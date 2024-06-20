# 提取图像特征
# 参考自：https://www.kaggle.com/christofhenkel/extract-image-features-from-pretrained-nn
# 并进行了一些修改
# 如果使用多张图片，我们使用PCA获取最重要的特征

# 4. 图像处理

# 4.a) 预处理（调整大小，加载图像）
pca = PCA(n_components=1, random_state=1234)
img_size = 256  # 图像尺寸
train_ids = train['PetID'].values  # 训练集的宠物ID
test_ids = test['PetID'].values  # 测试集的宠物ID


def resize_to_square(im):
    old_size = im.shape[:2]  # 获取原图像的尺寸 (高度, 宽度)
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])  # 计算新的尺寸
    im = cv2.resize(im, (new_size[1], new_size[0]))  # 调整图像大小
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]  # 边界填充颜色为黑色
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


def load_image(path):
    image = cv2.imread(path)
    new_image = resize_to_square(image)  # 调整图像为固定尺寸
    new_image = preprocess_input(new_image)  # 预处理图像数据
    return new_image


# 4.b) 构建模型（DenseNet 121）
inp = Input((256, 256, 3))
backbone = DenseNet121(input_tensor=inp,
                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",
                       include_top=False)
x = backbone.output
x = GlobalAveragePooling2D()(x)  # 全局平均池化
x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)  # 添加一维维度
x = AveragePooling1D(4)(x)  # 一维平均池化
out = Lambda(lambda x: x[:, :, 0])(x)  # 提取一维特征

m = Model(inp, out)  # 创建模型

########################################################################################################################
# 4.c) 图像特征提取

max_using_images = 5  # 每个宠物最多使用的图片数量


def extract_img_features(mode, files, pet_ids, set_tmp):
    features_h = {}
    features_a = {}

    for pet_id in tqdm(pet_ids):
        photoAmt = int(set_tmp.loc[pet_id, 'PhotoAmt'])  # 获取当前宠物的照片数量
        if photoAmt == 0:
            dim = 1
            batch_images_m = np.zeros((1, img_size, img_size, 3))  # 初始化一个单通道的图片数组
        else:
            dim = min(photoAmt, max_using_images)  # 确保使用的图片数量不超过最大值
            batch_images_m = np.zeros((dim, img_size, img_size, 3))  # 初始化一个数组存储图片
            try:
                urls = files[pet_id]
                for i, u in enumerate(urls[:dim]):
                    try:
                        batch_images_m[i] = load_image(u)  # 加载图片
                    except:
                        pass
            except:
                pass

        batch_preds_m = m.predict(batch_images_m)  # 对图片进行预测
        pred = pca.fit_transform(batch_preds_m.T)  # 使用PCA降维
        features_a[pet_id] = pred.reshape(-1)  # 存储降维后的特征
        features_h[pet_id] = batch_preds_m[0]  # 存储原始特征

    feats_h = pd.DataFrame.from_dict(features_h, orient='index')  # 转换为DataFrame格式
    feats_a = pd.DataFrame.from_dict(features_a, orient='index')  # 转换为DataFrame格式

    return feats_h, feats_a
