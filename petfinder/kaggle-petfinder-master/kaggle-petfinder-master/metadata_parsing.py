# 从 Wojtek Rosinski 的 kernel 分支 https://www.kaggle.com/wrosinski/baselinemodeling
# 添加情感和元数据特征

class PetFinderParser(object):

    def __init__(self, debug=False):
        self.debug = debug
        self.sentence_sep = ' '  # 句子分隔符

        # 因为主数据框已经包含描述，所以不需要提取
        self.extract_sentiment_text = True

    def open_metadata_file(self, filename):
        """
        加载元数据文件。
        """
        with open(filename, 'r') as f:
            metadata_file = json.load(f)
        return metadata_file

    def open_sentiment_file(self, filename):
        """
        加载情感文件。
        """
        with open(filename, 'r') as f:
            sentiment_file = json.load(f)
        return sentiment_file

    def open_image_file(self, filename):
        """
        加载图像文件。
        """
        image = np.asarray(Image.open(filename))
        return image

    def parse_sentiment_file(self, file):
        """
        解析情感文件。输出包含情感特征的数据框。
        """
        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = self.sentence_sep.join(file_entities)

        if self.extract_sentiment_text:
            file_sentences_text = [x['text']['content'] for x in file['sentences']]
            file_sentences_text = self.sentence_sep.join(file_sentences_text)
        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        file_sentences_sentiment = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()

        file_sentiment.update(file_sentences_sentiment)

        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
        if self.extract_sentiment_text:
            df_sentiment['text'] = file_sentences_text

        df_sentiment['entities'] = file_entities
        df_sentiment = df_sentiment.add_prefix('sentiment_')

        return df_sentiment

    def parse_metadata_file(self, file):
        """
        解析元数据文件。输出包含元数据特征的数据框。
        """
        file_keys = list(file.keys())

        if 'labelAnnotations' in file_keys:
            # file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']) * 0.5)]
            file_annots = file['labelAnnotations'][:]
            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_top_score = np.nan
            file_top_desc = ['']

        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']

        file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
        file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

        file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()

        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
        else:
            file_crop_importance = np.nan

        df_metadata = {
            'annots_score': file_top_score,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'annots_top_desc': self.sentence_sep.join(file_top_desc)
        }

        df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T
        df_metadata = df_metadata.add_prefix('metadata_')

        return df_metadata


# 辅助函数，用于并行数据处理：
def extract_additional_features(pet_id, mode='train'):
    sentiment_filename = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    metadata_filenames = sorted(
        glob.glob('../input/petfinder-adoption-prediction/{}_metadata/{}*.json'.format(mode, pet_id)))
    if len(metadata_filenames) > 0:
        for f in metadata_filenames:
            metadata_file = pet_parser.open_metadata_file(f)
            df_metadata = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]

    return dfs


########################################################################################################################
# 2.c 解析数据

def pet_parsing(train, test):
    # 从训练集和测试集中获取唯一的宠物ID：
    debug = False
    train_pet_ids = train.PetID.unique()
    test_pet_ids = test.PetID.unique()

    if debug:
        train_pet_ids = train_pet_ids[:10]
        test_pet_ids = test_pet_ids[:5]

    # 训练集：
    # 并行处理数据：
    dfs_train = Parallel(n_jobs=6, verbose=1)(
        delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)

    # 提取处理后的数据并将其格式化为数据框：
    train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
    train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]

    train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
    train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

    print(train_dfs_sentiment.shape, train_dfs_metadata.shape)

    # 测试集：
    # 并行处理数据：
    dfs_test = Parallel(n_jobs=6, verbose=1)(
        delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)

    # 提取处理后的数据并将其格式化为数据框：
    test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
    test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]

    test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
    test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

    print(test_dfs_sentiment.shape, test_dfs_metadata.shape)
    return train_dfs_sentiment, train_dfs_metadata, test_dfs_sentiment, test_dfs_metadata


pet_parser = PetFinderParser()
train_dfs_sentiment, train_dfs_metadata, test_dfs_sentiment, test_dfs_metadata = pet_parsing(train, test)
