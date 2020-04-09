class TransferLearningConfiguration():
    def __init__(self, input_model_path, output_model_path, input_smiles_path, save_every_n_epochs=1, batch_size=128,
                 clip_gradient_norm=1., num_epochs=10, starting_epoch=1, shuffle_each_epoch=True,
                 collect_stats_frequency=1, adaptive_lr_config=None, validation_smiles_path=None,
                 standardize=True, randomize=False):
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.input_smiles_path = input_smiles_path
        self.save_every_n_epochs = max(0, save_every_n_epochs)
        self.batch_size = max(0, batch_size)
        self.clip_gradient_norm = max(0.0, clip_gradient_norm)
        self.num_epochs = max(num_epochs, 1)
        self.starting_epoch = max(starting_epoch, 1)
        self.shuffle_each_epoch = shuffle_each_epoch
        self.collect_stats_frequency = collect_stats_frequency
        self.adaptive_lr_config = adaptive_lr_config
        self.validation_smiles_path = validation_smiles_path
        self.standardize = standardize
        self.randomize = randomize
