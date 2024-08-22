import os
import fasttext


def train(data_path, model_path):
    model = fasttext.train_unsupervised(
        input=data_path,
        model='skipgram',  # 또는 'cbow'
        lr=0.05,
        dim=100,
        ws=5,
        epoch=300,
        minCount=3,
        verbose=2
    )
    model.save_model(model_path)



if __name__ == '__main__':
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'otc_model_lower.bin')
    train('data/otc_lower.txt', model_path)

