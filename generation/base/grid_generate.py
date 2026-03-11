import hydra
from omegaconf import DictConfig


from generation.base import grid_schema




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config : DictConfig)->None:
    grid_schema.init_globals(config)
    print(grid_schema.MAX_GRID_WIDTH)


    # in here add data generation code


if __name__ == "__main__":
    main()
