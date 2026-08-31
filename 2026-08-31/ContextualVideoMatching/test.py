from dataset import get_mock_data
from model import ContextualVideoMatcher, retrieve_videos


def test_retrieval_pipeline():
    _, articles, videos = get_mock_data()
    model = ContextualVideoMatcher()
    results = retrieve_videos(model, articles[0], videos, threshold=-1.0, top_k=2)
    assert len(results) == 2
    assert all(video.language == articles[0].language for video, _ in results)
    print("ContextualVideoMatching toy pipeline test passed")


if __name__ == "__main__":
    test_retrieval_pipeline()
