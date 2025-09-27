import argparse
from nlmt_v2.recommend import recommend

def main():
    parser = argparse.ArgumentParser(description="Get language recommendations.")
    parser.add_argument("known_languages", nargs="+", help="List of languages you already know.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return.")
    args = parser.parse_args()

    recommendations = recommend(args.known_languages, args.top_k)
    print(recommendations)