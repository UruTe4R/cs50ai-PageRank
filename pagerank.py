import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    n_pages = len(corpus.keys()) # number of pages in corpus
    links = corpus[page] # set of links from page
    n_links = len(links) # number of links from page

    # if page has no links, return equl probability distribution
    if n_links == 0:
        probability = 1 / n_pages
        return {page: probability for page in corpus.keys()}
    
    p_to_next_page = damping_factor / n_links
    p_to_random_page = round((1 - damping_factor) / (len(corpus[page]) + 1), 10)

    probability = {link: round(p_to_next_page + p_to_random_page, 10) for link in links}
    probability[page] = p_to_random_page
    return probability


    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # pick first page randomly
    first_page = random.choice(list(corpus.keys()))
    p = transition_model(corpus, first_page, damping_factor)
    # set counter
    ranks = {page: 0 for page in corpus.keys()}

    # i == 0
    chosen_page = random.choices(list(p.keys()), weights=p.values(), k=1)[0]
    ranks[chosen_page] += 1

    for _ in range(n - 1):
        p = transition_model(corpus, chosen_page, damping_factor)
        chosen_page = random.choices(list(p.keys()), weights=p.values(), k=1)[0]
        ranks[chosen_page] += 1
    
    return {page: (count / n) for page, count in ranks.items()}
        



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    d = damping_factor
    N = len(corpus.keys()) # Number of pages in corpus
    convergence_threshold = 0.001

    ranks = {page: 1 / N for page in corpus.keys()} # set initial ranks

    # create dict of incoming links
    incoming_links = {page: set() for page in corpus.keys()}
    for key, value in corpus.items():
        # if there are no outgoing links from "key", add key to each incoming link
        if len(value) == 0:
            for page in incoming_links.keys():
                incoming_links[page].add(key)
            continue
        for link in value:
            incoming_links[link].add(key)

    # calculate Pagerank for each page
    while True:
        new_ranks = ranks.copy()
        for page, links_from in incoming_links.items():
            total = []
            for link_from in links_from:
                numlinks = len(corpus[link_from])
                if numlinks == 0:
                    numlinks = N
                total.append(ranks[link_from] / numlinks)
            
            a = np.array(total)
            new_ranks[page] = (1 - d) / N + d * np.sum(a)

        # check for convergence
        if all(abs(rank - ranks[page]) < convergence_threshold for page, rank in new_ranks.items()):
            break
        ranks = new_ranks

    return ranks




if __name__ == "__main__":
    main()
