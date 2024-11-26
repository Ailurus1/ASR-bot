from .sber import SberDataset

DATASETS = {
    "sber-golos-farfield": SberDataset("bond005/sberdevices_golos_100h_farfield"),
    "sber-golos-crowd": SberDataset("bond005/sberdevices_golos_10h_crowd"),
}
