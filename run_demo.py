from __future__ import annotations

from src.grounded_qa import answer_question


def main() -> None:
    corpus = [
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 1. Business",
            "text": "NVCT operates a cloud analytics platform that enables customers to monitor and analyze operational performance.",
            "source_url": "https://example.com/filing",
            "document_name": "filing.htm",
        },
        {
            "ticker": "NVCT",
            "form": "10-K",
            "filing_date": "2024-12-31",
            "section": "Item 7A. Market Risk",
            "text": "The company is exposed to changes in interest rates and customer concentration risk.",
            "source_url": "https://example.com/filing",
            "document_name": "filing.htm",
        },
    ]

    supported = answer_question("What is NVCT's business?", corpus)
    unsupported = answer_question("What is the company's exact lawsuit settlement amount?", corpus)

    print("SUPPORTED:")
    print(supported)
    print()
    print("UNSUPPORTED:")
    print(unsupported)


if __name__ == "__main__":
    main()
