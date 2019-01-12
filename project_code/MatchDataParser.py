import json

EVENT_DATA_CONSTANT = 'events'


class MatchParser:
    def __init__(self, file_path):
        self.data = self.read_file(file_path)

    @staticmethod
    def read_file(file_path):
        with open(file_path) as f:
            data = json.load(f)
            return data

    def get_event(self, event_number=-1):
        num_of_events = len(self.data[EVENT_DATA_CONSTANT])
        if num_of_events == 1:
            return self.data[EVENT_DATA_CONSTANT][0]
        while not (event_number in range(num_of_events)):
            event_number = int(input('select an event between 0 and {}: '.format(num_of_events-1)))
        return self.data[EVENT_DATA_CONSTANT][event_number]


if __name__ == '__main__':
    path = "../match_data/event.json"
    a = MatchParser(path)
    event = a.get_event()
    a = 0
