from omegaconf import OmegaConf



config = OmegaConf.load("./configs/data_preparation_stage1.yaml")
print(type(config))
print(config['dataset']['target'])