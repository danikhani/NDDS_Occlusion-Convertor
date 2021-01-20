
from prepr_linemode import make_linemode_dataset
from visualize_dataset import vis_bb


def main():
    ########### small config
    # since the files are from cad and each one of them has a different length unit.
    # Use one of these values to scale the box: (0.01,0.1,1,10,100). The smaller value makes a bigger box
    scale = 1
    training_data_percentage = 0.9
    # folder of saved data
    dataset_folder = 'datasets/test'

    only_visulize = False
    visulize_frame = 0



    ###########################################################
    generated_folder = dataset_folder + '/generated'
    if only_visulize is not True:
        make_linemode_dataset(dataset_folder, generated_folder, scale, training_data_percentage)
    vis_bb(generated_folder, visulize_frame,None)


if __name__ == '__main__':
    main()