#include <iostream>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;

#include "lightnet.h"
#include "post_process.h"
#include "os_hal.h"
#include "VideoHelper.h"

#include "MiniTraceHelper.h"
#include "readerwriterqueue/readerwriterqueue.h"

#define CVUI_IMPLEMENTATION
#include "cvui/cvui.h"

//#define PARALLEL_GRAB_VIDEO
#define PARALLEL_RESIZE

using namespace moodycamel;

const char* params =
"{ h help ?     | false             | print usage          }"
"{ cfg          |openpose.cfg       | model configuration }"
"{ weights      |openpose.weight    | model weights }"
"{@source1      |0                  | source1 for processing   }"
"{@source2      |<none>             | source2 for processing (optional) }"
"{ width        | 0                 | width of video or camera device}"
"{ height       | 0                 | height of video or camera device}"
"{ g gui        | true              | show gui, press g to toggle }"
"{ f fullscreen | false             | show in fullscreen, press f to toggle }"
"{ fps          | 0                 | fps of video or camera device }"
"{ single_step  |                   | single step mode, press any key to move to next frame }"
"{ l loop       | true              | whether to loop the video}"
"{ video_pos    | 0                 | current position of the video file in milliseconds. }"
"{ player       | 1                 | current position for player, press p to toggle. }"
;

bool is_gui_visible = false;
bool is_fullscreen = false;

#define APP_NAME "Dancing Gaga"
#define VER_MAJOR 0
#define VER_MINOR 2
#define VER_PATCH 0

#define TITLE APP_NAME " " CVAUX_STR(VER_MAJOR) "." CVAUX_STR(VER_MINOR) "." CVAUX_STR(VER_PATCH)

cv::Rect dancer_rois[] =
{
    { 67,0, 240,360 },
    { 335,0, 240,360 },
};

#define CONCURRENT_PKT_COUNT 1

struct NetOutpus
{
    vector<float> net_outputs;
    Mat frame;
    int idx;
};

ReaderWriterQueue<NetOutpus> q_output(CONCURRENT_PKT_COUNT);

struct ControlPanel
{
    void setup()
    {
        cvui::init(WINDOW_NAME);
    }

    void update()
    {
        int x = 10;
        int y = 0;
        int dy_small = 16;
        int dy_large = 50;
        int width = 300;
        frame = cv::Scalar(49, 52, 49);

        cvui::text(frame, x, y += dy_large, "find_heatmap_peaks_thresh");
        cvui::trackbar(frame, x, y += dy_small, width, &find_heatmap_peaks_thresh, 0.0f, 1.0f);

        y += dy_small;

        cvui::text(frame, x, y += dy_large, "body_inter_min_above_th");
        cvui::trackbar(frame, x, y += dy_small, width, &body_inter_min_above_th, 0, 20);

        cvui::text(frame, x, y += dy_large, "body_inter_th");
        cvui::trackbar(frame, x, y += dy_small, width, &body_inter_th, 0.0f, 1.0f);

        cvui::text(frame, x, y += dy_large, "body_min_subset_cnt");
        cvui::trackbar(frame, x, y += dy_small, width, &body_min_subset_cnt, 0, 20);

        cvui::text(frame, x, y += dy_large, "body_min_subset_score");
        cvui::trackbar(frame, x, y += dy_small, width, &body_min_subset_score, 0.0f, 1.0f);

        y += dy_small;

        cvui::text(frame, x, y += dy_large, "render_thresh");
        cvui::trackbar(frame, x, y += dy_small, width, &render_thresh, 0.0f, 1.0f);

        y += dy_small;

        cvui::update();
        cv::imshow(WINDOW_NAME, frame);
    }

    const String WINDOW_NAME = "param";
    cv::Mat frame = cv::Mat(770, 350, CV_8UC3);

    // param
    float find_heatmap_peaks_thresh = 0.05;

    int body_inter_min_above_th = 9;
    float body_inter_th = 0.05;
    int body_min_subset_cnt = 6;
    float body_min_subset_score = 0.4;

    float render_thresh = 0.05;
};

int main(int argc, char **argv)
{
    MiniTraceHelper mr_hepler;
    ControlPanel param_window;

    CommandLineParser parser(argc, argv, params);
    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    auto cfg_path = parser.get<string>("cfg");
    auto weights_path = parser.get<string>("weights");

    if (cfg_path.find("body_25") != string::npos)
    {
        setPoseModel(op::PoseModel::BODY_25);
        cout << "setPoseModel BODY_25" << endl;
    }

    uint32_t NET_OUT_CHANNELS = getNetOutChannels();

    Mat frame;

    // 1. read args
    is_gui_visible = parser.get<bool>("gui");
    is_fullscreen = parser.get<bool>("fullscreen");
    Mat upscale_frame;

    int player = parser.get<int>("player");

    String sources[] = {
        parser.get<String>("@source1"),
        parser.get<String>("@source2"),
    };

    bool source_is_camera[] = {
        false,
        false,
    };

    VideoCapture captures[] = {
        safe_open_video(parser, sources[0], &source_is_camera[0]),
        safe_open_video(parser, sources[1], &source_is_camera[1]),
    };

    bool is_running = true;

    // 2. initialize net
    int net_inw = 0;
    int net_inh = 0;
    int net_outw = 0;
    int net_outh = 0;
    {
        MTR_SCOPE(__FILE__, "init_net");
        init_net(cfg_path.c_str(), weights_path.c_str(), &net_inw, &net_inh, &net_outw, &net_outh);
    }

    float scale = 0.0f;

    vector<float> net_inputs(net_inw * net_inh * 3);
    float* net_inputs_ptr = net_inputs.data();

    std::thread CUDA([&] {
        MTR_META_THREAD_NAME("2) CUDA");
        int frame_count = 0;

        Mat frames[2];
        Mat input_frame;
        Mat input_blob;

        while (is_running)
        {
            MTR_SCOPE_FUNC_I("frame", frame_count);

            {
                MTR_SCOPE(__FILE__, "pre process");

#ifdef PARALLEL_GRAB_VIDEO
                parallel_for_(Range(0, 2), [&](const Range& range) { for (int i = range.start; i < range.end; i++)
#else
                for (int i = 0; i < 2; i++)
#endif
                {
                    if (!safe_grab_video(captures[i], parser, frames[i], sources[i], source_is_camera[i]))
                    {
                        is_running = false;
                    }
                }
#ifdef PARALLEL_GRAB_VIDEO
                });
#endif

                if (!is_running) break;

                {
                    if (captures[1].isOpened())
                    {
                        MTR_SCOPE(__FILE__, "mix");

                        float ar = dancer_rois[player].width / (float)dancer_rois[player].height;

                        int crop_h = frames[0].rows;
                        int crop_w = ar * crop_h;
                        int crop_x = (frames[0].cols - crop_w) * 0.5f;
                        int crop_y = 0;
                        Rect src_crop{
                            crop_x, crop_y,
                            crop_w, crop_h,
                        };
                        Rect dst_crop = dancer_rois[player];
                        frame = frames[1];

                        resize(frames[0](src_crop), frame(dst_crop), dst_crop.size());
                    }
                    else
                    {
                        frame = frames[0];
                    }
                }

                {
                    MTR_SCOPE(__FILE__, "Mat to float*");

                    // 3. resize to net input size, put scaled image on the top left
                    create_netsize_im(frame, input_frame, net_inw, net_inh, &scale);
                    input_blob = dnn::blobFromImage(input_frame, 1.0f / 255, Size(net_inw, net_inh), Scalar(127, 127, 127), false);
                }
            }

            // 6. feed forward
            float *netoutdata = NULL;
            TickMeter tick;
            {
                MTR_SCOPE(__FILE__, "run_net");
                tick.start();
                netoutdata = run_net(input_blob);
                tick.stop();
                cout << "forward fee: " << tick.getTimeMilli() << "ms" << endl;
            }

            NetOutpus pkt;
            pkt.net_outputs = { netoutdata, netoutdata + net_outh*net_outw*NET_OUT_CHANNELS };
            pkt.idx = frame_count;
            pkt.frame = frame;
            q_output.try_emplace(pkt);

            frame_count++;
        }

        is_running = false;
    });

    std::thread postCUDA([&]() {
        int frame_count = 0;
        MTR_META_THREAD_NAME("3) post CUDA");

        vector<float> heatmap_peaks(3 * (POSE_MAX_PEOPLE + 1) * (NET_OUT_CHANNELS - 1));
        vector<float> heatmap(net_inw * net_inh * NET_OUT_CHANNELS);

        param_window.setup();

        while (is_running)
        {
            NetOutpus pkt;
            if (!q_output.try_dequeue(pkt))
                continue;

            MTR_SCOPE_FUNC_I("frame", pkt.idx);
            frame_count++;

            float* netoutdata = pkt.net_outputs.data();

            vector<float> keypoints;
            vector<int> keyshape;
            {
                MTR_SCOPE(__FILE__, "post process");

                // 7. resize net output back to input size to get heatmap
#ifdef PARALLEL_RESIZE
                parallel_for_(Range(0, NET_OUT_CHANNELS), [&](const Range& range) { for (int i = range.start; i < range.end; i++)
#else
                for (int i = 0; i < NET_OUT_CHANNELS; ++i)
#endif
                {
                    MTR_SCOPE(__FILE__, "resize");
                    Mat netout(net_outh, net_outw, CV_32F, (netoutdata + net_outh*net_outw*i));
                    Mat nmsin(net_inh, net_inw, CV_32F, heatmap.data() + net_inh*net_inw*i);
                    resize(netout, nmsin, Size(net_inw, net_inh), 0, 0, CV_INTER_CUBIC);
                }
#ifdef PARALLEL_RESIZE
                });
#endif
    // 8. get heatmap peaks
    find_heatmap_peaks(heatmap.data(), heatmap_peaks.data(), net_inw, net_inh, NET_OUT_CHANNELS, param_window.find_heatmap_peaks_thresh);

    // 9. link parts
    connect_bodyparts(keypoints, heatmap.data(), heatmap_peaks.data(), net_inw, net_inh,
        param_window.body_inter_min_above_th,
        param_window.body_inter_th,
        param_window.body_min_subset_cnt,
        param_window.body_min_subset_score,
        keyshape);
            }

        {
            MTR_SCOPE(__FILE__, "viz");
            // 10. draw result
            if (is_fullscreen)
            {
                if (upscale_frame.empty())
                {
                    int w, h;
                    getScreenResolution(w, h);
                    float ar = frame.cols / (float)frame.rows;
                    upscale_frame = Mat(h, h * ar, CV_8UC3);

                    setWindowProperty(TITLE, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
                    moveWindow(TITLE, 0, 0);
                }

                MTR_SCOPE(__FILE__, "upscale");
                resize(pkt.frame, upscale_frame, upscale_frame.size());
                pkt.frame = upscale_frame;
                scale = upscale_frame.cols / (float)net_inw;
            }
            render_pose_keypoints(pkt.frame, keypoints, keyshape, param_window.render_thresh, scale);

            {
                int num_persons = keyshape[0];
                int num_joints = keyshape[1];
                int dim_joint = keyshape[2];
                for (int person = 0; person < num_persons; ++person)
                {
                    for (int part = 0; part < num_joints; ++part)
                    {
                        int index = (person * num_joints + part) * dim_joint;
                        if (keypoints[index + 2] > param_window.render_thresh)
                        {
                            //Point center{keypoints[index] * scale, keypoints[index + 1] * scale};
                            //if (center.y < 200)
                            //{
                            //}
                        }
                    }
                }
            }

            // 11. show and save result
            {
                MTR_SCOPE(__FILE__, "imshow");
                cout << "people: " << keyshape[0] << endl;

                imshow(TITLE, pkt.frame);
                if (is_gui_visible)
                {
                    param_window.update();
                }
            }

            {
                MTR_SCOPE(__FILE__, "waitkey");
                int key = waitKey(1);
                if (key == 27) break;
                if (key == 'f') is_fullscreen = !is_fullscreen;
                if (key == 'g') is_gui_visible = !is_gui_visible;
                if (key == 'p') player = 1 - player;
            }
        }
        }
    is_running = false;
    });

    CUDA.join();
    postCUDA.join();

    return 0;
}
