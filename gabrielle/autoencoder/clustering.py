from tensorflow.keras.preprocessing.sequence import pad_sequences
from gabrielle.tokenizer import CharLevelTokenizer
from gabrielle.autoencoder.model import get_trained_encoder
from gabrielle.autoencoder.config import TextAutoConfig
from soyclustering import SphericalKMeans
from soyclustering import merge_close_clusters
from numpy import dot
from numpy.linalg import norm
from scipy.sparse import csr_matrix


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


class AESphericalKMeansClustering:
    def __init__(self, texts: list, n_clusters=100, max_iter=15):
        self.texts = list(set(texts))
        if len(texts) != len(self.texts):
            print(f"Got {len(texts)} texts but {len(self.texts)} unique data loaded.")
        self.tokenizer = CharLevelTokenizer().load_pretrained_tokenizer()
        self.autoencoder = get_trained_encoder(TextAutoConfig.SAVED_MODEL_PATH)
        self.clustering = SphericalKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            verbose=1,
            init='similar_cut',
            sparsity='minimum_df',
            minimum_df_factor=0.05
        )
        self.features = []
        self.cluster_ids = None
        self.cluster_centers = None

    def run(self, merge_clusters=False):
        self.features = self._get_text_features()
        self.cluster_ids = self.clustering.fit_predict(csr_matrix(self.features))
        self.cluster_centers = self.clustering.cluster_centers_
        if merge_clusters:
            print(f"Initial n_clusters={len(self.cluster_centers)}...", end=' ')
            self._merge_clusters()
            print(f" → Merged n_clusters={len(self.cluster_centers)}")

    def _get_text_features(self):
        tokenized = [x for x in self.tokenizer.encode(self.texts)]
        padded = pad_sequences(sequences=tokenized, maxlen=TextAutoConfig.MAX_LENGTH, padding='post')
        ndarray_vectors = self.autoencoder.predict(padded, batch_size=64, verbose=1)
        return ndarray_vectors

    def _merge_clusters(self):
        self.cluster_centers, self.cluster_ids = merge_close_clusters(
            self.cluster_centers,
            self.cluster_ids,
            max_dist=.5
        )

    def save_txt(self, filename=f'result.txt'):
        with open(filename, encoding='utf-8', mode='w') as result:
            state = ''
            for t, l in sorted(zip(self.texts, self.cluster_ids), key=lambda p: p[1]):
                cluster_center = self.cluster_centers[l]
                cluster_id = str(l).zfill(3)
                sent_feat = self.features[self.texts.index(t)]
                if state != cluster_id:  # Cluster ID가 달라지는 지점에서 줄바꿈
                    print(file=result)
                    state = cluster_id
                print(cluster_id + '\t' + '%0.8f' % cos_sim(sent_feat, cluster_center) + '\t' + t, file=result)


if __name__ == '__main__':
    # load data
    with open(TextAutoConfig.DOCUMENTS, encoding='utf-8') as f:
        data = f.read().splitlines()

    clustering = AESphericalKMeansClustering(data)
    clustering.run()
    clustering.save_txt()


