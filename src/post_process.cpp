#include "post_process.h"
#include "minitrace/minitrace.h"
#include "openpose/poseParameters.hpp"
#include "openpose/poseParametersRender.hpp"

using namespace std;
using namespace cv;
using namespace op;

template<typename T>
inline int intRound(const T a)
{
    return int(a + 0.5f);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

static PoseModel s_model = PoseModel::COCO_18;

void setPoseModel(PoseModel model)
{
    s_model = model;
}

uint32_t getNetOutChannels()
{
    // #define NET_OUT_CHANNELS 57 // 38 for pafs, 19 for parts
    return getPoseNumberBodyParts(s_model) + 1 + getPoseMapIndex(s_model).size();
}

void render_pose_keypoints(
    Mat &frame,
    const vector<float> &keypoints,
    vector<int> keyshape,
    const float threshold,
    float scale)
{
    MTR_SCOPE_FUNC();

    const int num_keypoints = keyshape[1];

    vector<unsigned int> pairs;
    vector<float> colors;
    if (s_model == PoseModel::BODY_25)
    {
        pairs = {
            POSE_BODY_25_PAIRS_RENDER_GPU
        };
        colors = {
            POSE_BODY_25_COLORS_RENDER_GPU
        };
    }
    else if (s_model == PoseModel::COCO_18)
    {
        pairs = {
            POSE_COCO_PAIRS_RENDER_GPU
        };
        colors = {
            POSE_COCO_COLORS_RENDER_GPU
        };
    }

    const int pairs_size = pairs.size();
    const int number_colors = colors.size();

    for (int person = 0; person < keyshape[0]; ++person)
    {
        // Draw lines
        for (int pair = 0u; pair < pairs_size; pair += 2)
        {
            const int index1 = (person * num_keypoints + pairs[pair]) * keyshape[2];
            const int index2 = (person * num_keypoints + pairs[pair + 1]) * keyshape[2];
            if (keypoints[index1 + 2] > threshold && keypoints[index2 + 2] > threshold)
            {
                const int color_index = pairs[pair + 1] * 3;
                Scalar color{colors[(color_index + 2) % number_colors],
                             colors[(color_index + 1) % number_colors],
                             colors[(color_index + 0) % number_colors]};
                Point keypoint1{intRound(keypoints[index1] * scale), intRound(keypoints[index1 + 1] * scale)};
                Point keypoint2{intRound(keypoints[index2] * scale), intRound(keypoints[index2 + 1] * scale)};
                line(frame, keypoint1, keypoint2, color, 2);
            }
        }
        // Draw circles
        for (int part = 0; part < num_keypoints; ++part)
        {
            const int index = (person * num_keypoints + part) * keyshape[2];
            if (keypoints[index + 2] > threshold)
            {
                const int color_index = part * 3;
                Scalar color{colors[(color_index + 2) % number_colors],
                             colors[(color_index + 1) % number_colors],
                             colors[(color_index + 0) % number_colors]};
                Point center{intRound(keypoints[index] * scale), intRound(keypoints[index + 1] * scale)};
                circle(frame, center, 3, color, -1);
            }
        }
    }
}

void connect_bodyparts(
    vector<float> &pose_keypoints,
    const float *const map,
    const float *const peaks,
    int mapw,
    int maph,
    const int inter_min_above_th,
    const float inter_th,
    const int min_subset_cnt,
    const float min_subset_score,
    vector<int> &keypoint_shape)
{
    MTR_SCOPE_FUNC();

    keypoint_shape.resize(3);

    vector<unsigned int> body_part_pairs = getPosePartPairs(s_model);
    vector<unsigned int> limb_idx = getPoseMapIndex(s_model);
    const int num_body_parts = getPoseNumberBodyParts(s_model);
    const int num_body_part_pairs = num_body_parts + 1;
    std::vector<std::pair<std::vector<int>, double>> subset;
    const int subset_counter_index = num_body_parts;
    const int peaks_offset = 3 * (POSE_MAX_PEOPLE + 1);
    const int map_offset = mapw * maph;

    for (unsigned int pair_index = 0u; pair_index < num_body_part_pairs; ++pair_index)
    {
        const int body_partA = body_part_pairs[2 * pair_index];
        const int body_partB = body_part_pairs[2 * pair_index + 1];
        const float *candidateA = peaks + body_partA * peaks_offset;
        const float *candidateB = peaks + body_partB * peaks_offset;
        const int nA = (int)(candidateA[0]); // number of part A candidates
        const int nB = (int)(candidateB[0]); // number of part B candidates

        // add parts into the subset in special case
        if (nA == 0 || nB == 0)
        {
            // Change w.r.t. other
            if (nA == 0) // nB == 0 or not
            {
                for (int i = 1; i <= nB; ++i)
                {
                    bool num = false;
                    for (unsigned int j = 0u; j < subset.size(); ++j)
                    {
                        const int off = body_partB * peaks_offset + i * 3 + 2;
                        if (subset[j].first[body_partB] == off)
                        {
                            num = true;
                            break;
                        }
                    }
                    if (!num)
                    {
                        std::vector<int> row_vector(num_body_part_pairs, 0);
                        // store the index
                        row_vector[body_partB] = body_partB * peaks_offset + i * 3 + 2;
                        // the parts number of that person
                        row_vector[subset_counter_index] = 1;
                        // total score
                        const float subsetScore = candidateB[i * 3 + 2];
                        subset.emplace_back(std::make_pair(row_vector, subsetScore));
                    }
                }
            }
            else // if (nA != 0 && nB == 0)
            {
                for (int i = 1; i <= nA; i++)
                {
                    bool num = false;
                    for (unsigned int j = 0u; j < subset.size(); ++j)
                    {
                        const int off = body_partA * peaks_offset + i * 3 + 2;
                        if (subset[j].first[body_partA] == off)
                        {
                            num = true;
                            break;
                        }
                    }
                    if (!num)
                    {
                        std::vector<int> row_vector(num_body_part_pairs, 0);
                        // store the index
                        row_vector[body_partA] = body_partA * peaks_offset + i * 3 + 2;
                        // parts number of that person
                        row_vector[subset_counter_index] = 1;
                        // total score
                        const float subsetScore = candidateA[i * 3 + 2];
                        subset.emplace_back(std::make_pair(row_vector, subsetScore));
                    }
                }
            }
        }
        else // if (nA != 0 && nB != 0)
        {
            std::vector<std::tuple<double, int, int>> temp;
            const int num_inter = 10;
            // limb PAF x-direction heatmap
            const float *const mapX = map + (num_body_part_pairs + limb_idx[2 * pair_index]) * map_offset;
            // limb PAF y-direction heatmap
            const float *const mapY = map + (num_body_part_pairs + limb_idx[2 * pair_index + 1]) * map_offset;
            // start greedy algorithm
            for (int i = 1; i <= nA; i++)
            {
                for (int j = 1; j <= nB; j++)
                {
                    const int dX = candidateB[j * 3] - candidateA[i * 3];
                    const int dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
                    const float norm_vec = float(std::sqrt(dX * dX + dY * dY));
                    // If the peaksPtr are coincident. Don't connect them.
                    if (norm_vec > 1e-6)
                    {
                        const float sX = candidateA[i * 3];
                        const float sY = candidateA[i * 3 + 1];
                        const float vecX = dX / norm_vec;
                        const float vecY = dY / norm_vec;
                        float sum = 0.;
                        int count = 0;
                        for (int lm = 0; lm < num_inter; lm++)
                        {
                            const int mX = fastMin(mapw - 1, intRound(sX + lm * dX / num_inter));
                            const int mY = fastMin(maph - 1, intRound(sY + lm * dY / num_inter));
                            const int idx = mY * mapw + mX;
                            const float score = (vecX * mapX[idx] + vecY * mapY[idx]);
                            if (score > inter_th)
                            {
                                sum += score;
                                ++count;
                            }
                        }

                        // parts score + connection score
                        if (count > inter_min_above_th)
                        {
                            temp.emplace_back(std::make_tuple(sum / count, i, j));
                        }
                    }
                }
            }
            // select the top minAB connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (!temp.empty())
            {
                std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());
            }
            std::vector<std::tuple<int, int, double>> connectionK;

            const int minAB = fastMin(nA, nB);
            // assuming that each part occur only once, filter out same part1 to different part2
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);
            int counter = 0;
            for (unsigned int row = 0u; row < temp.size(); row++)
            {
                const float score = std::get<0>(temp[row]);
                const int aidx = std::get<1>(temp[row]);
                const int bidx = std::get<2>(temp[row]);
                if (!occurA[aidx - 1] && !occurB[bidx - 1])
                {
                    // save two part score "position" and limb mean PAF score
                    connectionK.emplace_back(std::make_tuple(body_partA * peaks_offset + aidx * 3 + 2,
                                                             body_partB * peaks_offset + bidx * 3 + 2, score));
                    ++counter;
                    if (counter == minAB)
                    {
                        break;
                    }
                    occurA[aidx - 1] = 1;
                    occurB[bidx - 1] = 1;
                }
            }
            // Cluster all the body part candidates into subset based on the part connection
            // initialize first body part connection
            if (pair_index == 0)
            {
                for (const auto connectionKI : connectionK)
                {
                    std::vector<int> row_vector(num_body_parts + 3, 0);
                    const int indexA = std::get<0>(connectionKI);
                    const int indexB = std::get<1>(connectionKI);
                    const double score = std::get<2>(connectionKI);
                    row_vector[body_part_pairs[0]] = indexA;
                    row_vector[body_part_pairs[1]] = indexB;
                    row_vector[subset_counter_index] = 2;
                    // add the score of parts and the connection
                    const double subset_score = peaks[indexA] + peaks[indexB] + score;
                    subset.emplace_back(std::make_pair(row_vector, subset_score));
                }
            }
            // Add ears connections (in case person is looking to opposite direction to camera)
            else if (pair_index == 17 || pair_index == 18)
            {
                for (const auto &connectionKI : connectionK)
                {
                    const int indexA = std::get<0>(connectionKI);
                    const int indexB = std::get<1>(connectionKI);
                    for (auto &subsetJ : subset)
                    {
                        auto &subsetJ_first = subsetJ.first[body_partA];
                        auto &subsetJ_first_plus1 = subsetJ.first[body_partB];
                        if (subsetJ_first == indexA && subsetJ_first_plus1 == 0)
                        {
                            subsetJ_first_plus1 = indexB;
                        }
                        else if (subsetJ_first_plus1 == indexB && subsetJ_first == 0)
                        {
                            subsetJ_first = indexA;
                        }
                    }
                }
            }
            else
            {
                if (!connectionK.empty())
                {
                    for (unsigned int i = 0u; i < connectionK.size(); ++i)
                    {
                        const int indexA = std::get<0>(connectionK[i]);
                        const int indexB = std::get<1>(connectionK[i]);
                        const double score = std::get<2>(connectionK[i]);
                        int num = 0;
                        // if A is already in the subset, add B
                        for (unsigned int j = 0u; j < subset.size(); j++)
                        {
                            if (subset[j].first[body_partA] == indexA)
                            {
                                subset[j].first[body_partB] = indexB;
                                ++num;
                                subset[j].first[subset_counter_index] = subset[j].first[subset_counter_index] + 1;
                                subset[j].second = subset[j].second + peaks[indexB] + score;
                            }
                        }
                        // if A is not found in the subset, create new one and add both
                        if (num == 0)
                        {
                            std::vector<int> row_vector(num_body_part_pairs, 0);
                            row_vector[body_partA] = indexA;
                            row_vector[body_partB] = indexB;
                            row_vector[subset_counter_index] = 2;
                            const float subsetScore = peaks[indexA] + peaks[indexB] + score;
                            subset.emplace_back(std::make_pair(row_vector, subsetScore));
                        }
                    }
                }
            }
        }
    }

    // Delete people below thresholds, and save to output
    int number_people = 0;
    std::vector<int> valid_subset_indexes;
    valid_subset_indexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
    for (unsigned int index = 0; index < subset.size(); ++index)
    {
        const int subset_counter = subset[index].first[subset_counter_index];
        const double subset_score = subset[index].second;
        if (subset_counter >= min_subset_cnt && (subset_score / subset_counter) > min_subset_score)
        {
            ++number_people;
            valid_subset_indexes.emplace_back(index);
            if (number_people == POSE_MAX_PEOPLE)
            {
                break;
            }
        }
    }

    // Fill and return pose_keypoints
    keypoint_shape = {number_people, (int)num_body_parts, 3};
    if (number_people > 0)
    {
        pose_keypoints.resize(number_people * (int)num_body_parts * 3);
    }
    else
    {
        pose_keypoints.clear();
    }
    for (unsigned int person = 0u; person < valid_subset_indexes.size(); ++person)
    {
        const auto &subsetI = subset[valid_subset_indexes[person]].first;
        for (int bodyPart = 0u; bodyPart < num_body_parts; bodyPart++)
        {
            const int base_offset = (person * num_body_parts + bodyPart) * 3;
            const int body_part_index = subsetI[bodyPart];
            if (body_part_index > 0)
            {
                pose_keypoints[base_offset] = peaks[body_part_index - 2];
                pose_keypoints[base_offset + 1] = peaks[body_part_index - 1];
                pose_keypoints[base_offset + 2] = peaks[body_part_index];
            }
            else
            {
                pose_keypoints[base_offset] = 0.f;
                pose_keypoints[base_offset + 1] = 0.f;
                pose_keypoints[base_offset + 2] = 0.f;
            }
        }
    }
}

void find_heatmap_peaks(
    const float *src,
    float *dst,
    const int SRCW,
    const int SRCH,
    const int SRC_CH,
    const float TH)
{
    MTR_SCOPE_FUNC();

    // find peaks (8-connected neighbor), weights with 7 by 7 area to get sub-pixel location and response
    const int SRC_PLANE_OFFSET = SRCW * SRCH;
    // add 1 for saving total people count, 3 for x, y, score
    const int DST_PLANE_OFFSET = (POSE_MAX_PEOPLE + 1) * 3;
    float *dstptr = dst;
    int c = 0;
    int x = 0;
    int y = 0;
    int i = 0;
    int j = 0;
    // TODO: reduce multiplication by using pointer
    for (c = 0; c < SRC_CH - 1; ++c)
    {
        int num_people = 0;
        for (y = 1; y < SRCH - 1 && num_people != POSE_MAX_PEOPLE; ++y)
        {
            for (x = 1; x < SRCW - 1 && num_people != POSE_MAX_PEOPLE; ++x)
            {
                int idx = y * SRCW + x;
                float value = src[idx];
                if (value > TH)
                {
                    const float TOPLEFT = src[idx - SRCW - 1];
                    const float TOP = src[idx - SRCW];
                    const float TOPRIGHT = src[idx - SRCW + 1];
                    const float LEFT = src[idx - 1];
                    const float RIGHT = src[idx + 1];
                    const float BUTTOMLEFT = src[idx + SRCW - 1];
                    const float BUTTOM = src[idx + SRCW];
                    const float BUTTOMRIGHT = src[idx + SRCW + 1];
                    if (value > TOPLEFT && value > TOP && value > TOPRIGHT && value > LEFT &&
                        value > RIGHT && value > BUTTOMLEFT && value > BUTTOM && value > BUTTOMRIGHT)
                    {
                        float x_acc = 0;
                        float y_acc = 0;
                        float score_acc = 0;
                        for (i = -3; i <= 3; ++i)
                        {
                            int ux = x + i;
                            if (ux >= 0 && ux < SRCW)
                            {
                                for (j = -3; j <= 3; ++j)
                                {
                                    int uy = y + j;
                                    if (uy >= 0 && uy < SRCH)
                                    {
                                        float score = src[uy * SRCW + ux];
                                        x_acc += ux * score;
                                        y_acc += uy * score;
                                        score_acc += score;
                                    }
                                }
                            }
                        }
                        x_acc /= score_acc;
                        y_acc /= score_acc;
                        score_acc = value;
                        dstptr[(num_people + 1) * 3 + 0] = x_acc;
                        dstptr[(num_people + 1) * 3 + 1] = y_acc;
                        dstptr[(num_people + 1) * 3 + 2] = score_acc;
                        ++num_people;
                    }
                }
            }
        }
        dstptr[0] = num_people;
        src += SRC_PLANE_OFFSET;
        dstptr += DST_PLANE_OFFSET;
    }
}

void create_netsize_im(
    const Mat &input,
    Mat &output,
    const int netw,
    const int neth,
    float *scale)
{
    MTR_SCOPE_FUNC();

    CV_Assert(!input.empty());

    // for tall image
    int newh = neth;
    float s = newh / (float)input.rows;
    int neww = input.cols * s;
    if (neww > netw)
    {
        //for fat image
        neww = netw;
        s = neww / (float)input.cols;
        newh = input.rows * s;
    }

    *scale = 1 / s;
    Rect dst_area(0, 0, neww, newh);
    if (output.empty())
    {
        output = Mat::zeros(neth, netw, CV_8UC3);
    }

    resize(input, output(dst_area), Size(neww, newh));
}
