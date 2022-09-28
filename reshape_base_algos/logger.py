# coding=utf-8
# summary:
# author: Jianqiang Ren
# date:
import logging


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }
    
    def __init__(self, filename, level='info',
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = logging.FileHandler(filename)
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger(filename='mylog.log', level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('warning')
    log.logger.error('error')
    log.logger.critical('critical')
