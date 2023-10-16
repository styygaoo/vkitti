from vkitti import vKITTIDataset
import os
from torch.utils.data import DataLoader




def main():
    height, width = 256, 512
    data_dir = r"C:\Users\gaoyi\Documents"
    n_samples = 1000

    traindir_source = os.path.join(data_dir, 'vkitti_data')
    source_split = os.path.join(os.getcwd(), 'Splits', 'vkitti')
    # print(source_split)
    source_split_file = os.path.join(source_split, 'vkitti_' + str(n_samples) + '_')



    source_trainset = vKITTIDataset(traindir_source, src_file=source_split_file + str(int(0)) + '.pickle',
                                    transform='valid', output_size=(height, width))

    src_loader = DataLoader(source_trainset, batch_size=4, shuffle=True, pin_memory=True, num_workers=8)
    for e in range(10):
        for i, (src_images, src_labels) in enumerate(src_loader):
            # rgb input
            print(i)
            print(src_labels.shape)
            print(src_images.shape)
            # input_d = torch.cat([src_images, trg_images])

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!


