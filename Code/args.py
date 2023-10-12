

class Args():
    def __init__(self):
        
        self.clean_tensorboard = False
        
        self.cuda = 1

        
        self.note = 'GraphRNN_VAE'

        self.graph_type = 'Twitter'


        self.max_num_node = None 
        self.max_prev_node = None 
        self.dropout_rate = 1
        self.exceed = 10000

        self.small_test = True

        if 'small' in self.graph_type:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
        self.hidden_size_rnn = int(64/self.parameter_shrink) 
        self.hidden_size_rnn_output = 15 
        self.low_embedding_size_rnn = int(128/self.parameter_shrink) 
        self.embedding_size_rnn_output = 128 

        self.batch_size = 16 
        self.test_batch_size = -999
        self.generate_size = 10
        self.num_layers = 2
        self.feature_numeric = True
        self.num_heads = None
        self.feature_number = None
        self.fixed_num = None

        
        self.num_workers = 4 
        self.batch_ratio = None 
        self.epochs = 3000 
        self.epochs_test_start = 1
        self.epochs_test = 1
        self.epochs_NMD_start = 10
        self.epochs_NMD = 10
        self.epochs_log = -999
        self.epochs_save_model = 100

        self.lr = 0.001
        self.milestones = [400, 1000]
        self.lr_rate = 0.01

        self.sample_time = 2 

        self.dir_input = "./"
        self.model_save_path = self.dir_input+'model_save/' 
        self.graph_save_path = self.dir_input+'graphs/'
        self.figure_save_path = self.dir_input+'figures/'
        self.timing_save_path = self.dir_input+'timing/'
        self.figure_prediction_save_path = self.dir_input+'figures_prediction/'
        self.nll_save_path = self.dir_input+'nll/'


        self.load = False 
        self.load_epoch = 3000
        self.save = True

        
        self.generator_baseline = 'BA'
        
        
        self.metric_baseline = 'clustering'

        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname_pred = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_train = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_test_'
        self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline+'_'+self.metric_baseline

