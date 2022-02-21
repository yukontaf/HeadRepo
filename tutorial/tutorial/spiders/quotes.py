import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            'https://www.coursera.org/learn/stats-for-data-analysis/lecture/5sogg/kak-priekrasny-mashinnoie-obuchieniie-i-analiz-dannykh',
             # 'https://www.coursera.org/learn/stats-for-data-analysis/home/week/2',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, cookies={'CSR3F-Token':'1646241764.pQHLORrxbGSjqqN3',
                                                                                  'CAUTH': 'Mlp3TW8wkCFtgA6kza02SePW34G1xz2_pn2pdruV3H1hide1UN3E5vSLDnRBmojR7J49GliXlN23VCoZP3MpUQ.YhrV8Zx4ZKnOwtJ849V-YA.waexRdooApEqIuZwkYkGmQlda2eszxlFilfDqAfrYHfh9vaGmS-Af-QvpyveQP4qTEGopVoeelwmv3hH5sFHr4GfIYcLkEdX8npbGxA61JWPpjyhNeLR6T0GgPLhykFvUJpZjbcTmVnLXZgl1XJzX8CCgb8AvdGQMFwB9gYjkZDrJIY5B3wEseB0PYB46820aHksqCZ1X3PeiphbcY5Lzz8x5eN_aIg7WEJ5wf7sfqH8a4zGpCp2xr0VMm79ILQJcPmvNw7zHeOPZoJhbIcT70Zojs-d7kObncsb3-Y8r6YVOfr2Ylq173DvsocDD-XxBggFpZoqE_W1Z-by_yac_sOYVvY_-B3Jkg68UyJ_sS5JExTS6D0AOviMKbfrZpiRZ7GM7rZu-m5ze4-RzKMDiSQOpooGNsXFU2_lM_pZIb4'})


    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f'quotes-{page}.html'
        with open(filename, 'wb') as f:
            f.write(response.xpath("//div[@class='rc-VideoMiniPlayer']").get())
        self.log(f'Saved file {filename}')