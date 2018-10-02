set OUTPUT=body_25
set PROTO=../openpose/models/pose/body_25/pose_deploy.prototxt
set MODEL=../openpose/models/pose/body_25/pose_iter_584000.caffemodel
REM python ../lightnet/modules/pytorch-caffe-darknet-convert/caffe2darknet.py %PROTO% %MODEL% %OUTPUT%.cfg %OUTPUT%.weights

set OUTPUT=coco
set PROTO=../openpose/models/pose/coco/pose_deploy_linevec.prototxt
set MODEL=../openpose/models/pose/coco/pose_iter_440000.caffemodel
REM python ../lightnet/modules/pytorch-caffe-darknet-convert/caffe2darknet.py %PROTO% %MODEL% %OUTPUT%.cfg %OUTPUT%.weights

set OUTPUT=mpi
set PROTO=../openpose/models/pose/mpi/pose_deploy_linevec.prototxt
set MODEL=../openpose/models/pose/mpi/pose_iter_160000.caffemodel
python ../lightnet/modules/pytorch-caffe-darknet-convert/caffe2darknet.py %PROTO% %MODEL% %OUTPUT%.cfg %OUTPUT%.weights
