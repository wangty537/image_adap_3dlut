import numpy as np
import torch

# if __name__ == "__main__":
    #trilinear_forward_pixel(lut, image, dim)
    # lut3d = np.random.randn(20, 3)
    #
    # grid = np.ones([2, 3])
    # grid = np.array([[0, 2, 1],
    #                  [8, 9, 4]])
    #
    # sel = lut3d[grid]
    # print(lut3d)
    # print(sel)
    # print(sel.shape)
    # lut3d = torch.from_numpy(lut3d)
    #
    # grid = torch.from_numpy(grid).long()
    #
    # sel = lut3d[grid]
    # print(lut3d)
    # print(sel)
    # print(sel.shape)




def trilinear_forward_pixel(lut, image, dim=33):
    height, width, channels = image.shape
    output_size = height * width
    binsize = 1.0000001 / (dim - 1)
    image = np.reshape(image, [-1, 3])
    lut = np.reshape(lut, [-1, 3])
    for index in range(output_size):
        r = image[index, 0]
        g = image[index, 1]
        b = image[index, 2]
        r_id = np.floor(r / binsize).astype(np.int32)
        g_id = np.floor(g / binsize).astype(np.int32)
        b_id = np.floor(b / binsize).astype(np.int32)

        r_d = np.fmod(r, binsize) / binsize
        g_d = np.fmod(g, binsize) / binsize
        b_d = np.fmod(b, binsize) / binsize

        id000 = r_id + g_id * dim + b_id * dim * dim
        id100 = r_id + 1 + g_id * dim + b_id * dim * dim
        id010 = r_id + (g_id + 1) * dim + b_id * dim * dim
        id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim
        id001 = r_id + g_id * dim + (b_id + 1) * dim * dim
        id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim
        id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim
        id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim

        w000 = (1 - r_d) * (1 - g_d) * (1 - b_d)
        w100 = r_d * (1 - g_d) * (1 - b_d)
        w010 = (1 - r_d) * g_d * (1 - b_d)
        w110 = r_d * g_d * (1 - b_d)
        w001 = (1 - r_d) * (1 - g_d) * b_d
        w101 = r_d * (1 - g_d) * b_d
        w011 = (1 - r_d) * g_d * b_d
        w111 = r_d * g_d * b_d

        image[index, 0] = w000 * lut[id000, 0] + w100 * lut[id100, 0] +\
                          w010 * lut[id010, 0] + w110 * lut[id110, 0] +\
                          w001 * lut[id001, 0] + w101 * lut[id101, 0] +\
                          w011 * lut[id011, 0] + w111 * lut[id111, 0]

        image[index, 1] = w000 * lut[id000, 1] + w100 * lut[id100, 1] +\
                          w010 * lut[id010, 1] + w110 * lut[id110, 1] +\
                          w001 * lut[id001, 1] + w101 * lut[id101, 1] +\
                          w011 * lut[id011, 1] + w111 * lut[id111, 1]

        image[index, 2] = w000 * lut[id000, 2] + w100 * lut[id100, 2] +\
                          w010 * lut[id010, 2] + w110 * lut[id110, 2] +\
                          w001 * lut[id001, 2] + w101 * lut[id101, 2] +\
                          w011 * lut[id011, 2] + w111 * lut[id111, 2]
        return image.reshape([height, width, channels])
def trilinear_forward(img, lut, lut_size=33):
    """
    :param img: h * w * channel numpy array, float
    :param lut: 3d lut, size[-1, 3]
    :param lut_size: default:33
    :return: img_lut
    """
    bin_size = 1.0000001 / (lut_size - 1)
    dim = lut_size
    # for i in range(h):
    #     for j in range(w):
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    r_id = np.floor(r / bin_size).astype(np.int32)
    g_id = np.floor(g / bin_size).astype(np.int32)
    b_id = np.floor(b / bin_size).astype(np.int32)
    r_d = np.fmod(r, bin_size) / bin_size
    g_d = np.fmod(g, bin_size) / bin_size
    b_d = np.fmod(b, bin_size) / bin_size

    id000 = r_id + g_id * dim + b_id * dim * dim
    id100 = r_id + 1 + g_id * dim + b_id * dim * dim
    id010 = r_id + (g_id + 1) * dim + b_id * dim * dim
    id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim
    id001 = r_id + g_id * dim + (b_id + 1) * dim * dim
    id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim
    id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim
    id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim

    w000 = (1 - r_d) * (1 - g_d) * (1 - b_d)
    w100 = r_d * (1 - g_d) * (1 - b_d)
    w010 = (1 - r_d) * g_d * (1 - b_d)
    w110 = r_d * g_d * (1 - b_d)
    w001 = (1 - r_d) * (1 - g_d) * b_d
    w101 = r_d * (1 - g_d) * b_d
    w011 = (1 - r_d) * g_d * b_d
    w111 = r_d * g_d * b_d

    w000 = w000[..., None]
    w100 = w100[..., None]
    w010 = w010[..., None]
    w110 = w110[..., None]
    w001 = w001[..., None]
    w101 = w101[..., None]
    w011 = w011[..., None]
    w111 = w111[..., None]
    rgb = w000 * lut[id000] + w100 * lut[id100] + \
        w010 * lut[id010] + w110 * lut[id110] + \
        w001 * lut[id001] + w101 * lut[id101] + \
        w011 * lut[id011] + w111 * lut[id111]

    return rgb

def trilinear_forward_tensor(img, lut, lut_size=33):
    """
    :param img: n * channel * h * w  tensor, float
    :param lut: 3d lut, size[-1, 3]
    :param lut_size: default:33
    :return: img_lut
    """
    lut = lut.reshape(3, -1).T
    #print(lut.shape)
    n, channel, h, w = img.shape
    img = img.permute([0, 2, 3, 1]).reshape(n * h, w, channel)
    bin_size = 1.0000001 / (lut_size - 1)
    dim = lut_size
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    r_id = torch.floor(r / bin_size).long()
    g_id = torch.floor(g / bin_size).long()
    b_id = torch.floor(b / bin_size).long()

    r_d = torch.fmod(r, bin_size) / bin_size
    g_d = torch.fmod(g, bin_size) / bin_size
    b_d = torch.fmod(b, bin_size) / bin_size

    id000 = r_id + g_id * dim + b_id * dim * dim
    id100 = r_id + 1 + g_id * dim + b_id * dim * dim
    id010 = r_id + (g_id + 1) * dim + b_id * dim * dim
    id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim
    id001 = r_id + g_id * dim + (b_id + 1) * dim * dim
    id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim
    id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim
    id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim
    #print(img.max().item(), r.max().item(), g.max().item(), b.max().item(), r_id.max().item(), g_id.max().item(), b_id.max().item(), id111.max().item())
    w000 = (1 - r_d) * (1 - g_d) * (1 - b_d)
    w100 = r_d * (1 - g_d) * (1 - b_d)
    w010 = (1 - r_d) * g_d * (1 - b_d)
    w110 = r_d * g_d * (1 - b_d)
    w001 = (1 - r_d) * (1 - g_d) * b_d
    w101 = r_d * (1 - g_d) * b_d
    w011 = (1 - r_d) * g_d * b_d
    w111 = r_d * g_d * b_d

    #print(id000.shape, id000.dtype, w000.shape, w000.dtype)
    #print(lut[id000].shape)
    w000 = w000[..., None]
    w100 = w100[..., None]
    w010 = w010[..., None]
    w110 = w110[..., None]
    w001 = w001[..., None]
    w101 = w101[..., None]
    w011 = w011[..., None]
    w111 = w111[..., None]

    rgb = w000 * lut[id000] + w100 * lut[id100] + \
        w010 * lut[id010] + w110 * lut[id110] + \
        w001 * lut[id001] + w101 * lut[id101] + \
        w011 * lut[id011] + w111 * lut[id111]
    rgb = rgb.reshape(n, h, w, channel).permute(0, 3, 1, 2)
    return rgb