from BASE_MODEL import BASE_RNN
import sys

#default parameter
FEATURE_SIZE = 106 # dataset input fields count
EMB_DIM = 32
BATCH_SIZE = 128
MAX_SEQ_LEN = 239
TRAING_STEPS = 10000000
STATE_SIZE = 128
GRAD_CLIP = 5.0
L2_NORM = 0.001
ADD_TIME = True
input_file="" #toy dataset
DNN_MODEL = True

if len(sys.argv) < 2:
    print("Please input learning rate. ex. 0.0001")
    sys.exit(0)

LR = float(sys.argv[1])
RUNNING_MODEL = BASE_RNN(EMB_DIM=EMB_DIM,
                         FEATURE_SIZE=FEATURE_SIZE,
                         BATCH_SIZE=BATCH_SIZE,
                         MAX_SEQ_LEN=MAX_SEQ_LEN,
                         TRAING_STEPS=TRAING_STEPS,
                         STATE_SIZE=STATE_SIZE,
                         LR=LR,
                         GRAD_CLIP=GRAD_CLIP,
                         L2_NORM=L2_NORM,
                         INPUT_FILE=input_file,
                         ADD_TIME_FEATURE=ADD_TIME,
                         FIND_PARAMETER=False,
                         DNN_MODEL=DNN_MODEL,
                         LOG_PREFIX="drsa")
RUNNING_MODEL.create_graph()
RUNNING_MODEL.run_model()
