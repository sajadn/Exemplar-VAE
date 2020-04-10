import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os


def imshow(img, title=None, interpolation=None, show_plot=False):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation=interpolation)
    if title is not None:
        plt.title(title)
    if show_plot:
        plt.show()


def generate_fancy_grid(config, dir, reference_data, generated, col_num=4, row_num=3):
    import cv2

    image_size = config.input_size[-1]
    width = col_num*image_size+2
    height = row_num*image_size+2

    print('references', reference_data.shape)
    print('generated', generated.shape)

    generated_dir = os.path.join(dir, 'generated/')
    os.makedirs(generated_dir, exist_ok=True)

    for k in range(len(reference_data)):
        grid = np.ones((config.input_size[0], height, width))
        original_image = reference_data[k].reshape(1, *config.input_size).cpu().detach().numpy()
        grid[:, 0:image_size, 0:image_size] = original_image
        generated_images = generated[k].reshape(-1, *config.input_size).cpu().detach().numpy()
        offset = 2
        counts = 0
        for i in range(row_num):
            j_counts = col_num
            extra_offset = 0
            if i == 0:
                j_counts = col_num-1
                extra_offset = image_size

            row = i*image_size+offset
            for j in range(j_counts):
                generated_images[counts]
                grid[:, row:row+image_size, extra_offset+j*image_size+offset:extra_offset+(j+1)*image_size+offset] = generated_images[counts]
                counts += 1

        if config.input_size[0] > 1:
            grid = np.transpose(grid, (1, 2, 0))
        grid = np.squeeze(grid)
        plt.imsave(arr=np.clip(grid, 0, 1),
                   fname=generated_dir + "generated_{}.png".format(k),
                   cmap='gray', format='png')

        img = cv2.imread(generated_dir + "generated_{}.png".format(k))
        res = cv2.resize(img, dsize=(width*3, height*3), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(generated_dir + "generated_{}.png".format(k), res)
    # plt.show()


def plot_images_in_line(images, args, dir, file_name):
    import cv2

    width = len(images) * 28
    height = 28
    grid = np.ones((height, width))
    for index, image in enumerate(images):
        image = image.reshape(*args.input_size).cpu().detach().numpy()
        grid[0:28, 28*index:28*(index+1)] = image[0]
    file_name = os.path.join(dir, file_name)
    plt.imsave(arr=grid / 255,
               fname=file_name,
               cmap='gray', format='png')

    img = cv2.imread(file_name)
    res = cv2.resize(img, dsize=(width*3, height*3), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(file_name, res)


def plot_images(config, x_sample, dir, file_name, size_x=3, size_y=3):
    if len(x_sample.shape) < 4:
        x_sample = x_sample.reshape(-1, *config.input_size)
    fig = plt.figure(figsize=(size_x, size_y))
    # fig = plt.figure(1)
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.01, hspace=0.01)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if config.input_type == 'binary' or config.input_type == 'gray':
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray')
        else:
            plt.imshow(sample)

    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)


