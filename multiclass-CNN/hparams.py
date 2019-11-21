import os
import torch

class Hparams():
    def __init__(self):

        self.cuda = True if torch.cuda.is_available() else False

        """
        Data Parameters
        """

        # os.makedirs('../input', exist_ok=True)
        os.makedirs('../model', exist_ok=True)
        # os.makedirs('../data/', exist_ok=True)
        os.makedirs('../results/', exist_ok=True)

        # self.train_csv = '/data1/amey/ASR-data/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/train.csv'
        # self.valid_csv = '/data1/amey/ASR-data/TAU-urban-acoustic-scenes-2019-development/evaluation_setup/valid.csv'
        # self.submit_csv = '/data1/amey/ASR-data/TAU-urban-acoustic-scenes-2019-leaderboard/evaluation_setup/test.csv'

        self.train_csv = '../evaluation_setup/fold1_train.csv'
        self.valid_csv = '../evaluation_setup/fold1_evaluate.csv'
        self.submit_csv = '../../../Downloads/ASR-data/TAU-urban-acoustic-scenes-2019-leaderboard/evaluation_setup/test.csv'

        self.dev_file = '../features/logmel_64frames_64melbins/TAU-urban-acoustic-scenes-2019-development.h5'
        self.submit_file = '../features/logmel_64frames_64melbins/TAU-urban-acoustic-scenes-2019-leaderboard.h5'

        self.scalar_file = '../scalars/logmel_64frames_64melbins/TAU-urban-acoustic-scenes-2019-development.h5'

        """
        Model Parameters
        """

        os.makedirs('../model/', exist_ok=True)

        self.input_shape = (640, 64)
        self.num_channel = 64
        self.num_classes = 10

        self.id_to_class = {
            0: 'airport',
            1: 'shopping_mall',
            2: 'metro_station',
            3: 'street_pedestrian',
            4: 'public_square',
            5: 'street_traffic',
            6: 'tram',
            7: 'bus',
            8: 'metro',
            9: 'park',
        }


        """
        Training parameters
        """

        self.gpu_device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        self.device_ids = [5]

        self.pretrained = False

        self.thresh = 0.5
        self.repeat_infer = 1

        self.num_epochs = 100
        self.batch_size = 16

        self.learning_rate = 0.00001

        self.momentum1 = 0.5
        self.momentum2 = 0.999

        self.avg_mode = 'micro'

        self.print_interval = 1000

        ################################################################################################################################################
        self.exp_name = 'multiclass-CNN-densenet/'
        self.dim3 = True
        ################################################################################################################################################

        self.result_dir = '../results/'+self.exp_name
        os.makedirs(self.result_dir, exist_ok=True)

        self.model_dir = '../model/' + self.exp_name
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = self.model_dir + 'model'


hparams = Hparams()
