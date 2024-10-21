import torch
import numpy as np
import matplotlib.pyplot as plt
from util.normImageTo255 import normImageTo255


def visImage(x, title):
    x2 = torch.squeeze(x, 0)
    x3 = x2.numpy()
    x4 = np.swapaxes(x3, 0, 2)
    x5 = normImageTo255(x4)
    plt.axis('off')
    plt.title(title)
    plt.imshow(x5.astype('uint8'))
    plt.show()


def show_images_rgb(images, cols=1, titles=None, pdf=[]):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """

    plt.rcParams.update({'font.size': 10})

    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)-1
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    rows = np.ceil(n_images / float(cols)) + 1

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        """
        x2 = torch.squeeze(image, 0)
        x3 = x2.numpy()
        x4 = np.swapaxes(x3, 0, 2)
        x4 = np.swapaxes(x4, 0, 1)
        x5 = normImageTo255(x4)
        """

        from skimage.transform import rescale, resize
        x6 = resize(image, (image.shape[0] * 2, image.shape[1] * 2),
                       anti_aliasing=True)

        if n == 0:
            a = fig.add_subplot(int(rows), int(cols), int(2), frameon=False)
        else:
            a = fig.add_subplot(int(rows), int(cols), int(3 + n))
        if image.ndim == 2:
            plt.gray()
        # plt.imshow(x6.astype('uint8'))
        plt.imshow(x6)
        a.set_title(title, fontsize=10)
        a.set_axis_off()

    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.axis('off')

    for ax in fig.axes:
        ax.axis("off")

    # plt.savefig('ciao.png')
    # plt.show()

    # save the current figure
    pdf.savefig(fig)

    # close figure
    plt.close(fig)


def show_images_gcam(images, cols=1, titles=None, pdf=[]):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """

    plt.rcParams.update({'font.size': 10})

    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)-1
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    rows = np.ceil(n_images / float(cols))

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        """
        x2 = torch.squeeze(image, 0)
        x3 = x2.numpy()
        x4 = np.swapaxes(x3, 0, 2)
        x4 = np.swapaxes(x4, 0, 1)
        x5 = normImageTo255(x4)
        """

        from skimage.transform import rescale, resize
        x6 = resize(image, (image.shape[0] * 2, image.shape[1] * 2),
                       anti_aliasing=True)

        match n:
            case 0:
                pass
                """
                a = fig.add_subplot(int(rows), int(cols), 1, frameon=False)
                plt.imshow(x6)
                a.set_title(title, fontsize=10)
                a.set_axis_off()
                """
            case 1:
                a = fig.add_subplot(int(rows), int(cols), 4, frameon=False)
                plt.imshow(x6)
                a.set_title(title, fontsize=10)
                a.set_axis_off()
            case 2:
                a = fig.add_subplot(int(rows), int(cols), 2, frameon=False)
                plt.imshow(x6)
                a.set_title(title, fontsize=10)
                a.set_axis_off()
            case 3:
                a = fig.add_subplot(int(rows), int(cols), 5, frameon=False)
                plt.imshow(x6)
                a.set_title(title, fontsize=10)
                a.set_axis_off()
            case 4:
                pass
                """
                a = fig.add_subplot(int(rows), int(cols), 3, frameon=False)
                plt.imshow(x6)
                a.set_title(title, fontsize=10)
                a.set_axis_off()
                """
            case 5:
                a = fig.add_subplot(int(rows), int(cols), 6, frameon=False)
                plt.imshow(x6)
                a.set_title(title, fontsize=10)
                a.set_axis_off()

        if image.ndim == 2:
            plt.gray()
        # plt.imshow(x6.astype('uint8'))

    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.axis('off')

    # plt.show()

    for ax in fig.axes:
        ax.axis("off")

    # plt.savefig('ciao.png')
    # plt.show()

    # save the current figure
    pdf.savefig(fig)

    # close figure
    plt.close(fig)
