from torch.utils.data import Dataset
from model import Article, Video, tokenize, synthesize_hypothetical_video_metadata


class MatchingDataset(Dataset):
    def __init__(self, samples: list) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        article, positive_video = self.samples[index]
        return tokenize(synthesize_hypothetical_video_metadata(article)), tokenize(f"{positive_video.title}. {positive_video.description}")


def get_mock_data():
    articles = [
        Article("New creator product review rules", "A marketplace explains updated creator review policy for beauty products and refund claims.", "en"),
        Article("Live shopping trend report", "Short video creators are driving discovery for outdoor products through demonstrations and comparisons.", "en"),
        Article("治理公告", "平台加强达人内容违规检测，重点治理夸大宣传和低质搬运。", "zh"),
    ]
    videos = [
        Video("v1", "Creator review policy explained", "Beauty ecommerce compliance, refunds, disclosure and trust signals.", "en", 0.8, 0.7),
        Video("v2", "Outdoor gear creator comparison", "Short video demonstration for outdoor ecommerce shopping discovery.", "en", 0.7, 0.9),
        Video("v3", "达人内容治理解读", "电商达人违规检测、夸大宣传识别、低质内容清洗。", "zh", 0.9, 0.8),
    ]
    return list(zip(articles, videos)), articles, videos
