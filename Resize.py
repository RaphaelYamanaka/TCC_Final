import cv2
import os


def make_square(im, sqr_size, fill_color=[0, 0, 0]):
    y, x = im.shape[0:2]
    if y > sqr_size or x > sqr_size:
        if y > x:
            div = sqr_size / y
            y *= div
            x *= div
        else:
            div = sqr_size / x
            y *= div
            x *= div
    size = max(sqr_size, x, y)
    add_x = int((size - x)/2)
    add_y = int((size - y)/2)
    im = cv2.resize(im, (int(x), int(y)))
    new_im = cv2.copyMakeBorder(
        im, add_y, add_y, add_x, add_x, cv2.BORDER_CONSTANT, value=fill_color)
    return new_im


if __name__ == "__main__":
    data_dir = './Cars/Teste_Carros/resizedCars'  # Data
    data_resized_dir = "./Cars/Teste_Carros/Photo_car"  # Resized data

    IMG_SIZE = 256
    BLACK = [0, 0, 0]
    desenho = False

    for each in os.listdir(data_dir):
        image = cv2.imread(os.path.join(data_dir, each))
        image = make_square(image, IMG_SIZE, BLACK)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        if desenho:
            nome = each[0:-19] + str(IMG_SIZE) + "x" + \
                str(IMG_SIZE) + "_desenho.png"
        else:
            nome = each[0:-11] + str(IMG_SIZE) + "x" + str(IMG_SIZE) + ".png"

        cv2.imwrite(os.path.join(data_resized_dir, nome), image)
