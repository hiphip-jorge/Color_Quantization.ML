import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
import imageio
import os

# import images
# img1 - strawberries
# img2 - vegetation and waterfall
# img3 - football
img1 = imageio.imread('./Images/image2.jpg')
img2 = imageio.imread('./Images/image3.jpg')
img3 = imageio.imread('./Images/image5.jpg')
# path for clustered results
path = "./quantizedImages/"

def quantize(raster, k):
    width, height, depth = raster.shape
    reshaped_raster = np.reshape(raster, (width * height, depth))

    model = cluster.KMeans(n_clusters=k)
    labels = model.fit_predict(reshaped_raster)
    palette = model.cluster_centers_

    quantized_raster = np.reshape(palette[labels], (width, height, palette.shape[1]))

    return quantized_raster


def plot(original, img1, img2, img3):
    # figure initialization and size
    fig = plt.figure(figsize=(8, 8))
    # num of columns and rows
    columns = 4
    rows = 3
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        # plot images
        # original - 1, 5, 9
        if i == 1:
            plt.imshow(original[0])
            plt.title('original')
        if i == 5:
            plt.imshow(original[1])
            plt.title('original')
        if i == 9:
            plt.imshow(original[2])
            plt.title('original')
        # img1 - 2-4
        if i == 2:
            plt.imshow(img1[0].astype(np.uint8))
            plt.title('k = 4')
        if i == 3:
            plt.imshow(img1[1].astype(np.uint8))
            plt.title('k = 8')
        if i == 4:
            plt.imshow(img1[2].astype(np.uint8))
            plt.title('k = 16')
        # img2 - 6-8
        if i == 6:
            plt.imshow(img2[0].astype(np.uint8))
            plt.title('k = 4')
        if i == 7:
            plt.imshow(img2[1].astype(np.uint8))
            plt.title('k = 8')
        if i == 8:
            plt.imshow(img2[2].astype(np.uint8))
            plt.title('k = 16')
        # img3 - 10-12
        if i == 10:
            plt.imshow(img3[0].astype(np.uint8))
            plt.title('k = 4')
        if i == 11:
            plt.imshow(img3[1].astype(np.uint8))
            plt.title('k = 8')
        if i == 12:
            plt.imshow(img3[2].astype(np.uint8))
            plt.title('k = 16')
    plt.show()


def quantize_set(img):
    img_set = []
    # 8, 16, 32 clusters
    for i in range(2, 5):
        img_set.append(quantize(img, 2**i))
    return img_set


def main():
    original = [img1, img2, img3]
    img1_set = quantize_set(img1)
    img2_set = quantize_set(img2)
    img3_set = quantize_set(img3)
    plot(original, img1_set, img2_set, img3_set)

    # place images in folder
    for i in range(3):
        imageio.imwrite(path+"img1_cluster{}.jpg".format(2**(i+2)), img1_set[i].astype(np.uint8))
        imageio.imwrite(path+"img2_cluster{}.jpg".format(2**(i+2)), img2_set[i].astype(np.uint8))
        imageio.imwrite(path+"img3_cluster{}.jpg".format(2**(i+2)), img3_set[i].astype(np.uint8))

    # compare original size with quantized
    print("Original sizes:")
    print("Size of strawberries img is {}KB".format(int(os.path.getsize("./Images/image2.jpg")/1000)))
    print("Size of waterfall img is {}KB".format(int(os.path.getsize("./Images/image3.jpg")/1000)))
    print("Size of football img is {}KB".format(int(os.path.getsize("./Images/image5.jpg")/1000)))

    # store quantized sizes
    size = []
    for i in range(3):
        size.append("Size of strawberries with {} clusters is {}KB".format(2**(i+2), int(os.path.getsize(path+"img1_cluster{}.jpg".format(2**(i+2)))/1000)))
    for i in range(3):
        size.append("Size of waterfall with {} clusters is {}KB".format(2**(i+2), int(os.path.getsize(path+"img2_cluster{}.jpg".format(2**(i+2)))/1000)))
    for i in range(3):
        size.append("Size of football with {} clusters is {}KB".format(2**(i+2), int(os.path.getsize(path+"img3_cluster{}.jpg".format(2**(i+2)))/1000)))

    print("\nSizes after quantization:")
    for i in range(len(size)):
        if i % 3 == 0:
            print()
        print(size[i])


if __name__ == '__main__':
    main()
