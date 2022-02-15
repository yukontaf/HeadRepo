import scrapy


class RlsspiderSpider(scrapy.Spider):
    name = 'rlsspider'
    allowed_domains = ['https://www.rlsnet.ru/']
    start_urls = ['http://https://www.rlsnet.ru//']

    def parse(self, response):
        pass
