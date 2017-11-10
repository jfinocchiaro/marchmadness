#!/usr/bin/env python3

import re
import urllib
#from crawler import Crawler, CrawlerCache
import requests
import pickle
#print(content)

class Crawl():
    def __init__(self, url):
        self.url = url
        self.header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
        self.data = {}

    def load_data(self):
        self.data = pickle.load(open("kenpom_data.p", "rb"))
        self.id = len(self.data)
        #print(self.id)

    def dump_data(self):
        pickle.dump(self.data, open("kenpom_data.p", "wb"))

    def crawl_teams(self):
        response = requests.get(self.url, headers=self.header)
        content = response.content.decode('utf-8')

        print(content)

        # matches = re.findall('<a href="([^ ]*)">(.*)<\/a>', content)
        # for match in matches:
        #     if (match is not None) & (len(match[1]) > 0):
        #         if ("?" in match[0]) | ("?" in match[1]):
        #             continue
        #
        #         href = str(match[0])
        #         name = str(match[1])
        #
        #         if name not in self.artists:
        #             self.artists[name] = {}
        #             self.artists[name]["href"] = href
        #             self.artists[name]["id"] = self.id
        #             self.id += 1
        #             #print("href: %s\t| name: %s" % (match[0], match[1]))

    def print_data(self):
        for team in self.data:
            print(self.data[team])

def main(depth=10):
        year = 2002
        kenPomCrawler = Crawl("https://kenpom.com/index.php?y=" + str(year))

        # Load old data
        #kenPomCrawler.load_artists()

        # Initialize data
        kenPomCrawler.crawl_teams()
        #print("Gathering data...")

        # Check
        kenPomCrawler.print_data()

        # Dump data for use later
        kenPomCrawler.dump_data()

if __name__ == '__main__':
    main(depth=10)
