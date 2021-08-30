#
# validate.py
#
import utils.binvox_rw
from utils.augmentations import SSDAugmentation
from data import *
import time
import torch
import torch.backends.cudnn as cudnn
from ssd import build_ssd
from torch.autograd import Variable
from pathlib import Path
import warnings
import os
import glob

warnings.simplefilter("ignore", UserWarning)
import pickle
from PIL import Image

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_gt_information(filename):
    retarr = np.zeros((0, 7))
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            items = row[0].split(',')
            retarr = np.insert(retarr, 0, np.asarray(items), 0)

    retarr[:, 0:6] = retarr[:, 0:6] * 1000

    return retarr


def load_pretrained_model(file_weights):
    #
    ssd_net = build_ssd(cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

    ssd_net.load_weights(file_weights)

    return net.cuda() if (torch.cuda.is_available()) else net.cpu()


def tensor_to_float(val):
    if val < 0:
        val = 0

    if val > 1:
        val = 1

    return float(val)


def rotate_sample(sample, rotation, reverse=False):
    if reverse:
        if rotation == 1:
            sample = np.rot90(sample, -2, (0, 1)).copy()
        elif rotation == 2:
            sample = np.rot90(sample, -1, (0, 1)).copy()
        elif rotation == 3:
            sample = np.rot90(sample, -1, (1, 0)).copy()
        elif rotation == 4:
            sample = np.rot90(sample, -1, (2, 0)).copy()
        elif rotation == 5:
            sample = np.rot90(sample, -1, (0, 2)).copy()
    else:
        if rotation == 1:
            sample = np.rot90(sample, 2, (0, 1)).copy()
        elif rotation == 2:
            sample = np.rot90(sample, 1, (0, 1)).copy()
        elif rotation == 3:
            sample = np.rot90(sample, 1, (1, 0)).copy()
        elif rotation == 4:
            sample = np.rot90(sample, 1, (2, 0)).copy()
        elif rotation == 5:
            sample = np.rot90(sample, 1, (0, 2)).copy()

    return sample


def soft_nms_pytorch(boxes, box_scores, sigma=0.5):
    # short explanation for NMS == Non-Maximum Suppression (NMS)
    # https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab

    dets = boxes[:, 0:6].copy() * 1000

    N = dets.shape[0]

    indexes = torch.arange(0, N, dtype=torch.double).view(N, 1).cpu()
    dets = torch.from_numpy(dets).double().cpu()
    scores = torch.from_numpy(box_scores.copy()).double().cpu()

    dets = torch.cat((dets, indexes), dim=1).cpu()

    z1 = dets[:, 0]
    y1 = dets[:, 1]
    x1 = dets[:, 2]
    z2 = dets[:, 3]
    y2 = dets[:, 4]
    x2 = dets[:, 5]
    # scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)

    for i in range(N):
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        zz1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        yy1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        zz2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 4].to("cpu").numpy(), dets[pos:, 4].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 5].to("cpu").numpy(), dets[pos:, 5].to("cpu").numpy())

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        l = np.maximum(0.0, zz2 - zz1 + 1)
        inter = torch.tensor(w * h * l).cpu()
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter)).cpu()

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma).cpu()

        scores[pos:] = weight * scores[pos:]

    # print(scores)
    max_margin = 0
    thresh = 0
    for i in range(scores.shape[0] - 1):
        if scores[i] - scores[i + 1] > max_margin:
            max_margin = scores[i] - scores[i + 1]
            thresh = (scores[i] + scores[i + 1]) / 2

    # thresh = (scores[1] + scores[2])/2

    keep = dets[:, 6][scores > thresh].int()

    # print(keep.shape)

    return keep.to("cpu").numpy()


def get_predicted_information(filename, net, folder_stl):
    #
    with open(filename + '.binvox', 'rb') as f:
        model = utils.binvox_rw.read_as_3d_array(f).data

    transform = SSDAugmentation(cfg['min_dim'], MEANS, phase='test')

    images = []

    # create 6 PNGs
    for rotation in range(6):
        raw_img, _ = create_img(model, rotation)

        img, _, _ = transform(raw_img, 0, 0)

        images.append(img)

        # print("saving Images here")
        new_filename = filename + "_" + str(rotation) + ".png"
        cv2.imwrite(new_filename, raw_img)
        # new_filename = filename + "_" + str(rotation) + "_result.png"
        # cv2.imwrite(new_filename, raw_img)
        """raw_img = Image.open(new_filename)
        raw_img = raw_img.resize((1000, 1000), Image.ANTIALIAS)
        raw_img.save(new_filename)"""

    images = torch.tensor(images).permute(0, 3, 1, 2).float()

    images = images.cuda() if (torch.cuda.is_available()) else images.cpu()

    images = Variable(images)

    out = net(images, 'test')
    out.cuda() if (torch.cuda.is_available()) else out.cpu()

    boxes_for_visuali = np.zeros((0, 9))
    cur_boxes = np.zeros((0, 9))

    for i in range(6):

        for j in range(out.shape[1]):
            label = out[i, j, 1].detach().cpu()

            if label == 0:
                continue

            score = out[i, j, 0].detach().cpu()

            x1 = tensor_to_float(out[i, j, 2])
            y1 = tensor_to_float(out[i, j, 3])
            x2 = tensor_to_float(out[i, j, 4])
            y2 = tensor_to_float(out[i, j, 5])
            z1 = 0.0
            z2 = tensor_to_float(out[i, j, 6])

            if x1 >= x2 or y1 >= y2 or z2 <= 0:
                continue

            a = z1
            b = y1
            c = x1
            d = z2
            e = y2
            f = x2

            boxes_for_visuali = np.append(boxes_for_visuali,
                                          np.array([a, b, c, d, e, f, label - 1, score, i]).reshape(1, 9), axis=0)

            # print("why is this necessary when working with validation")
            # Converting local coordinates to global coordinates for later comparison with .csv
            if i == 1:
                a = 1 - z2
                b = 1 - y2
                c = x1
                d = 1 - z1
                e = 1 - y1
                f = x2
            elif i == 2:
                a = y1
                b = 1 - z2
                c = x1
                d = y2
                e = 1 - z1
                f = x2
            elif i == 3:
                a = 1 - y2
                b = z1
                c = x1
                d = 1 - y1
                e = z2
                f = x2
            elif i == 4:
                a = 1 - x2
                b = y1
                c = z1
                d = 1 - x1
                e = y2
                f = z2
            elif i == 5:
                a = x1
                b = y1
                c = 1 - z2
                d = x2
                e = y2
                f = 1 - z1

            cur_boxes = np.append(cur_boxes, np.array([a, b, c, d, e, f, label - 1, score, i]).reshape(1, 9), axis=0)

    keepidx = soft_nms_pytorch(cur_boxes[:, :7], cur_boxes[:, -1])
    cur_boxes = cur_boxes[keepidx, :]
    cur_boxes[:, 0:6] = 10000 * cur_boxes[:, 0:6]

    keepidx_2 = soft_nms_pytorch(boxes_for_visuali[:, :7], boxes_for_visuali[:, -1])
    boxes_for_visuali = boxes_for_visuali[keepidx_2, :]
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(boxes_for_visuali, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cur_boxes


def get_lvec(labels):
    # input = [4,6,2,5,5]
    # output = [0,0,1,0,1,2,1,0,.....]
    results = np.zeros(24)

    for i in labels:
        results[int(i)] += 1

    return results.astype(int)


def eval_metric(prediction_labels, true_labels, positives):
    #
    precision = divide_arrs(positives, prediction_labels)
    recall = divide_arrs(positives, true_labels)

    return precision, recall


# 0/0 => 1
# x/0 => 0
def divide_arrs(arr1, arr2):
    ret = []

    for idx in range(len(arr2)):
        if (arr2[idx] == arr1[idx]):
            ret.append(1)
        else:
            if arr2[idx] == 0:
                ret.append(0)
            else:
                ret.append(arr1[idx] / arr2[idx])

    return ret


def cal_detection_performance(predicted_information, true_information):
    #
    prediction_labels = get_labels(predicted_information)

    true_labels = get_labels(true_information)

    positives = np.minimum(true_labels, prediction_labels)

    return prediction_labels, true_labels, positives


def get_labels(information):
    #
    boxes_labels = information[:, 6]
    labels = get_lvec(boxes_labels)

    return labels


def test_ssdnet(folder_stl, file_weights):
    #
    net = load_pretrained_model(file_weights)
    metric = cal_detection_performance

    predictions = np.zeros(24)
    truelabels = np.zeros(24)
    truepositives = np.zeros(24)

    with torch.no_grad():
        with open(os.devnull, 'w') as devnull:
            for filename in Path(folder_stl).glob('*.STL'):
                filename = str(filename).replace('.STL', '')

                # TODO split large pictures (Showcase large)

                predicted_information = get_predicted_information(filename, net, folder_stl)

                ground_truth_information = get_gt_information(filename + '.csv')

                prediction_label, true_label, positive = metric(predicted_information, ground_truth_information)

                predictions += prediction_label
                truelabels += true_label
                truepositives += positive

                print(filename)

                print(true_label)
                print(prediction_label)
                print(positive)

    # print("THIS METRIC IS WRONG ")
    precision, recall = eval_metric(predictions, truelabels, truepositives)
    print('Precision scores')
    precision = np.mean(precision)
    print(precision)
    print('Recall scores')
    recall = np.mean(recall)
    print(recall)
    print('F scores')
    print((2 * recall * precision) / (recall + precision))


def visualize(folder_stl):
    predictions_list = [f for f in os.listdir(folder_stl) if f.endswith('.pickle')]

    for predicton_container in predictions_list:

        with open(folder_stl + predicton_container, 'rb') as handle:
            predictions = pickle.load(handle)

        picture = folder_stl + predicton_container.replace(".pickle", "")

        counter = 0
        for x in range(len(predictions)):

            data = predictions[x]
            selected_image = int(data[8])

            z1 = int(data[0] * 64)
            x1 = int(data[1] * 64)
            y1 = int(data[2] * 64)
            z2 = int(data[3] * 64) - 1
            x2 = int(data[4] * 64) - 1
            y2 = int(data[5] * 64) - 1
            Feature = data[6]
            prop = data[7]

            if prop >= 0.5:

                selected_image = picture + "_" + str(selected_image) + ".png"
                im = np.array(Image.open(selected_image))
                im = cv2.imread(selected_image)

                print("found feature " + str(Feature) + " in picture " + selected_image)

                color = {
                    0: [255, 255, 0, 255],
                    1: [255, 0, 0, 255],
                    2: [0, 255, 0, 255],
                    3: [0, 0, 255, 255],
                    4: [255, 127, 0, 255],
                    5: [255, 212, 0, 255],
                    6: [255, 255, 0, 255],
                    7: [191, 255, 0, 255],
                    8: [106, 255, 0, 255],
                    9: [0, 234, 255, 255],
                    10: [0, 149, 255, 255],
                    11: [0, 64, 255, 255],
                    12: [170, 0, 255, 255],
                    13: [255, 0, 170, 255],
                    14: [237, 185, 185, 255],
                    15: [231, 233, 185, 255],
                    16: [185, 237, 224, 255],
                    17: [185, 215, 237, 255],
                    18: [220, 185, 237, 255],
                    19: [143, 35, 35, 255],
                    20: [143, 106, 3, 255],
                    21: [79, 143, 35, 255],
                    22: [35, 98, 143, 255],
                    23: [107, 35, 143, 255],
                    24: [115, 115, 115, 255],
                    25: [204, 204, 204, 255]
                }[Feature]

                color = color[0:3][::-1]
                im[x1][y1] = color
                im[x2][y2] = color

                for x in range(x2 - x1):
                    im[x1 + x][y1] = color
                    im[x1 + x][y2] = color

                for x in range(y2 - y1):
                    im[x1][y1 + x] = color
                    im[x2][y1 + x] = color

                # backup_filename = selected_image.replace(".png", ("_" + str(counter) + ".png"))
                cv2.imwrite(selected_image, im)
                counter += 1

def create_weigths():
    #
    file_weights = 'weights/VOC.pth'
    flag = os.path.isfile(file_weights)
    if (flag):
        return

    import zipfile

    zips = glob.glob('weights/VOC.zip.*')  # os.listdir("weights/VOC.zip.*")
    target = os.path.relpath("weights/voc.zip")
    for zipName in zips:
        source = zipName
        with open(target, "ab") as f:
            with open(source, "rb") as z:
                f.write(z.read())

    zip_ref = zipfile.ZipFile(target, "r")
    zip_ref.extractall("weights")


def run():
    start_time = time.time()

    create_weigths()

    folder_stl = 'data/MulSet/set20/'
    file_weights = 'weights/VOC.pth'

    files = glob.glob(folder_stl + '/*.png', recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    test_ssdnet(folder_stl, file_weights)

    visualize(folder_stl)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    run()
