import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = ['WERKS', 'SCENARIO', 'KTOKK', 'VSTATU', 'VPATD', 'EKORG', 'EKGRP',
                        'TOTGRQTY', 'TOTIRQTY', 'NODLGR', 'NODLIR', 'DIFGRIRD', 'DIFGRIRV',
                        'STATUS']
LABEL_COLUMN = 'STATUS'
DEFAULTS = [['ML01'],['3'], ['1'], ['1'], [30.0], ['1'], ['A'], [0.], [80.0], [0.], [90.0], [-80.0], [-38100.0], [1]]

INPUT_FILE = None  # set from task.py
EVAL_FILE = None

# Define some hyperparameters
EVAL_INTERVAL = 45
TRAIN_STEPS = 10000
EVAL_STEPS = None
BATCH_SIZE = 512

# Define your feature columns
def create_feature_cols():
#   lat_buck = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('latitude'), 
#                                                  boundaries = np.arange(32.0, 42, 1).tolist())
#   long_buck = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('longitude'),
#                                                   boundaries = np.arange(1, 52, 1).tolist())
    werks_c = tf.feature_column.categorical_column_with_vocabulary_list(
            key='WERKS',
            vocabulary_list=['ML01','ML02','ML03'])
    scenario_c = tf.feature_column.categorical_column_with_vocabulary_list(
            key='SCENARIO',
            vocabulary_list=['1','2','3','4'])
    ktokk_c = tf.feature_column.categorical_column_with_vocabulary_list(
            key='KTOKK',
            vocabulary_list=['1','2'])    
    vstatu_c = tf.feature_column.categorical_column_with_vocabulary_list(
            key='VSTATU',
            vocabulary_list=['1','2'])
    ekorg_c = tf.feature_column.categorical_column_with_vocabulary_list(
            key='EKORG',
            vocabulary_list=['1','2'])   
    ekgrp_c = tf.feature_column.categorical_column_with_vocabulary_list(
            key='EKGRP',
            vocabulary_list=['A','B','C'])

    return [
        tf.feature_column.indicator_column(werks_c),
        tf.feature_column.indicator_column(scenario_c),
        tf.feature_column.indicator_column(ktokk_c),
        tf.feature_column.indicator_column(vstatu_c),
        tf.feature_column.indicator_column(ekorg_c),
        tf.feature_column.indicator_column(ekgrp_c),
        tf.feature_column.numeric_column('VPATD'),
        tf.feature_column.numeric_column("TOTGRQTY"),
        tf.feature_column.numeric_column("TOTIRQTY"),
        tf.feature_column.numeric_column("NODLGR"),
        tf.feature_column.numeric_column("NODLIR"),
        tf.feature_column.numeric_column("DIFGRIRD"),
        tf.feature_column.numeric_column("grminusirbyvpatd"),
        tf.feature_column.numeric_column("difgrirdbytotgrqty"),
        tf.feature_column.numeric_column("DIFGRIRV")
  ]

#Data reader
def read_dataset(file_pattern, mode, batch_size = 512):    
    def _input_fn(v_test=False):
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label
        
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(file_pattern)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        
        #Begins - Uncomment for testing only -----------------------------------------------------<
        if v_test == True:
            with tf.Session() as sess:
                print(sess.run(batch_features))
        #End - Uncomment for testing only -----------------------------------------------------<
        return batch_features, batch_labels
    return _input_fn

# Serving function for external call
def serving_fn():
    feature_placeholders  = {'WERKS' : tf.placeholder(tf.string, [None]),
            'SCENARIO' : tf.placeholder(tf.string, [None]),
            'KTOKK' : tf.placeholder(tf.string, [None]),
            'VSTATU' : tf.placeholder(tf.string, [None]),
            'EKORG' : tf.placeholder(tf.string, [None]),
            'EKGRP' : tf.placeholder(tf.string, [None]),
            'VPATD' : tf.placeholder(tf.float32, [None]),
            'TOTGRQTY' : tf.placeholder(tf.float32, [None]),
            'TOTIRQTY' : tf.placeholder(tf.float32, [None]),
            'NODLGR' : tf.placeholder(tf.float32, [None]),
            'NODLIR' : tf.placeholder(tf.float32, [None]),
            'DIFGRIRD' : tf.placeholder(tf.float32, [None]),
            'DIFGRIRV' : tf.placeholder(tf.float32, [None])
    }

    #Features with transformation logic
    features = {
                key: tf.expand_dims(tensor, -1)
                for key, tensor in feature_placeholders.items()
            }
    
    #feat_changed = add_engineered(features.copy())
    return tf.estimator.export.ServingInputReceiver(add_engineered(features), feature_placeholders )

#Feature engineering: Tf version of add_new_features
def add_new_features_tf(df_temp):   
    #Add any feature engineering or new column here
    df_temp['grminusirbyvpatd'] = ( df_temp['TOTGRQTY'] - df_temp['TOTIRQTY'] ) / df_temp['VPATD']
    
    df_temp['difgrirdbytotgrqty'] = tf.where( tf.not_equal(tf.cast(df_temp['TOTGRQTY'], tf.float32), tf.cast(0, tf.float32)),
                                              tf.cast(tf.divide(df_temp['DIFGRIRD'], df_temp['TOTGRQTY']), tf.float32),
                                              tf.cast(tf.zeros_like(df_temp['DIFGRIRD']), tf.float32))
    return df_temp

#RETURNS a pre-processing( New columns + Transformations ) function to be used inside pipeline
def create_add_engineered_fn():
    #Pass all the above calculated values to be used by main function which will be called in Pipeline
    def fn_add_engineered(features):       
        #Add new features AGAIN as the function add_engineered will be called with data
        features = add_new_features_tf(features)
        return features
    
    return fn_add_engineered

add_engineered = create_add_engineered_fn()

# Create estimator train and evaluate function
def train_and_evaluate(output_dir):    
##### Create Canned estimator instance
    ## setting the checkpoint interval to be much lower for this task
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL, 
                                        keep_checkpoint_max = 3)
    
    estimator = tf.estimator.DNNClassifier(feature_columns=create_feature_cols(),
                                          model_dir = output_dir,
                                          n_classes=2,
                                          hidden_units=[32,64,64,64,64,64,
                                                        64,64,64,64,64,64,
                                                        64,64,64,64,64,64,
                                                        32],
                                          dropout = 0.2,
                                          optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                          config = run_config)
    train_spec = tf.estimator.TrainSpec(input_fn = read_dataset(
                                                file_pattern = INPUT_FILE,
                                                mode = tf.estimator.ModeKeys.TRAIN,
                                                batch_size = BATCH_SIZE),
                                      max_steps = TRAIN_STEPS)
    exp = tf.estimator.LatestExporter("decision", serving_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn = read_dataset(
                                                file_pattern = EVAL_FILE,
                                                mode = tf.estimator.ModeKeys.EVAL,
                                                batch_size = 128),
                                    steps = EVAL_STEPS, 
                                    exporters = exp,
                                    start_delay_secs = 20, # start evaluating after N seconds, 
                                    throttle_secs = EVAL_INTERVAL)  # evaluate every N seconds
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)