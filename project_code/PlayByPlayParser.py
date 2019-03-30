import csv


class EventParser:
    def __init__(self, file_path):
        self.row_names = list()
        self.events = list()
        self.read_file(file_path)

    def read_file(self, file_path):
        with open(file_path) as f:
            data = csv.reader(f, delimiter=',')
            line_count = 0
            for row in data:
                if line_count == 0:
                    self.row_names = row
                    line_count += 1
                else:
                    self.events.append(PlayByPlayEvent(row))
                    line_count += 1


class PlayByPlayEvent:
    def __init__(self, row):
        self.original_row = row


if __name__ == '__main__':
    e = EventParser('../match_data/0021500491.csv')
    a = 0
