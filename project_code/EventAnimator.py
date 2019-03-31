import matplotlib.pyplot as plt
from matplotlib import animation

from project_code import MatchDataParser
from project_code import PlayByPlayParser

VISITOR = 'visitor'
HOME = 'home'
BALL = -1
NAME = 'name'
TEAM_ID = 'teamid'
PLAYERS = 'players'
PLAYER_NAME = 'player_name'
PLAYER_FIRSTNAME = 'firstname'
PLAYER_LASTNAME = 'lastname'
PLAYER_NUMBER = 'jersey'
PLAYER_POSITION = 'position'
PLAYER_ID = 'playerid'
MOMENTS = 'moments'
X_POSITION = 'xpos'
Y_POSITION = 'ypos'
Z_POSITION = 'zpos'
SHOTCLOCK = 'shotclock'
COORDINATES = 'coords'
COLOR = 'color'
POSITION_COORDINATES = 5

X_MIN = 0
X_MAX = 94
Y_MIN = 0
Y_MAX = 50

base_path = '../match_data/'


class EventAnimation:
    def __init__(self, event, play_by_play):
        self.event = event
        self.team_info = self.get_team_details()
        self.event_frames = self.get_event_frames()
        self.play_by_play = play_by_play
        self.pbp_events = []
        self.show_event()
        self.last_time = -1

    def get_team_details(self):
        info = {}
        event = self.event
        info[event[VISITOR][TEAM_ID]] = {NAME: event[VISITOR][NAME], COLOR: '#A50044', PLAYERS: {}}
        info[event[HOME][TEAM_ID]] = {NAME: event[HOME][NAME], COLOR: '#004D98', PLAYERS: {}}

        for player in event[VISITOR][PLAYERS]:
            player_data = {PLAYER_NAME: "{0} {1}".format(player[PLAYER_FIRSTNAME], player[PLAYER_LASTNAME]),
                           PLAYER_NUMBER: player[PLAYER_NUMBER], PLAYER_POSITION: player[PLAYER_POSITION]}
            info[event[VISITOR][TEAM_ID]][PLAYERS][player[PLAYER_ID]] = player_data

        for player in event[HOME][PLAYERS]:
            player_data = {PLAYER_NAME: "{0} {1}".format(player[PLAYER_FIRSTNAME], player[PLAYER_LASTNAME]),
                           PLAYER_NUMBER: player[PLAYER_NUMBER], PLAYER_POSITION: player[PLAYER_POSITION]}
            info[event[HOME][TEAM_ID]][PLAYERS][player[PLAYER_ID]] = player_data

        info[BALL] = {NAME: "Ball", COLOR: '#FA8320', PLAYERS: {-1: {NAME: "Ball", PLAYER_NUMBER: "", PLAYER_POSITION: ""}}}
        return info

    def get_event_frames(self):
        moments = self.event[MOMENTS]
        frames = []
        for moment in moments:
            player_position = []
            for p in moment[POSITION_COORDINATES]:
                player_position.append({TEAM_ID: p[0], PLAYER_ID: p[1],
                                        X_POSITION: p[2], Y_POSITION: p[3], Z_POSITION: p[4]})
            frame = {SHOTCLOCK: moment[2], COORDINATES: player_position, 'period': moment[0]}
            frames.append(frame)
        return frames

    def get_pbp_events(self, frames):
        events = []
        period = frames[0]['period']
        start_time = frames[0][SHOTCLOCK]
        self.last_time = start_time
        end_time = frames[-1][SHOTCLOCK]
        for i in range(len(play_by_play.events)):
            e = play_by_play.display_pbp(i)
            if int(e['period']) == period and int(start_time + 1) >= e['time_left'] >= int(end_time):
                events.append(e)
        return events

    def update_positions(self, i, player_circles, player_annotations, pbp):
        if i == 0 and len(self.pbp_events) > 0:
            self.last_time = 13*60
            pbp[0].set_text('Home event: ' + self.pbp_events[0]['home_desc'])
            pbp[1].set_text('Neutral event: ' + self.pbp_events[0]['neutral_desc'])
            pbp[2].set_text('Away event: ' + self.pbp_events[0]['away_desc'])
        frame = self.event_frames[i]
        player_coordinates = frame[COORDINATES]
        for j, circle in enumerate(player_circles):
            circle.center = player_coordinates[j][X_POSITION], player_coordinates[j][Y_POSITION]
            player_annotations[j].set_position(circle.center)
        player_circles[0].radius = player_coordinates[0][Z_POSITION]/7
        for p in self.pbp_events:
            if self.last_time > p['time_left'] >= int(frame[SHOTCLOCK]):
                self.last_time = p['time_left']
                pbp[0].set_text('Home event: ' + p['home_desc'])
                pbp[1].set_text('Neutral event: ' + p['neutral_desc'])
                pbp[2].set_text('Away event: ' + p['away_desc'])
                break
        return player_circles

    def show_event(self):
        axs = plt.axes(ylim=(Y_MIN, Y_MAX), xlim=(X_MIN, X_MAX))
        axs.axis('off')
        axs.grid(False)
        frames = self.event_frames
        team_info = self.team_info
        self.pbp_events = self.get_pbp_events(frames)
        start_frame = frames[0]
        figure = plt.gcf()

        player_annotations = []
        for player in start_frame[COORDINATES]:
            player_annotations.append(
                axs.annotate(team_info[player[TEAM_ID]][PLAYERS][player[PLAYER_ID]][PLAYER_NUMBER], xy=[0, 0],
                             color='w', horizontalalignment='center',
                             verticalalignment='center', fontweight='bold'))

        player_circles = [plt.Circle((0, 0), 12/7, color=team_info[player[TEAM_ID]][COLOR])
                          for player in start_frame[COORDINATES]]

        for circle in player_circles:
            axs.add_patch(circle)

        pbp_desc = []
        if len(self.pbp_events) > 0:
            pbp_desc.append(plt.figtext(0.3, 0.12, 'Home event: ' + self.pbp_events[0]['home_desc']))
            pbp_desc.append(plt.figtext(0.3, 0.07, 'Neutral event: ' + self.pbp_events[0]['neutral_desc']))
            pbp_desc.append(plt.figtext(0.3, 0.02, 'Away event: ' + self.pbp_events[0]['away_desc']))

        anim = animation.FuncAnimation(
            figure, self.update_positions,
            fargs=(player_circles, player_annotations, pbp_desc),
            frames=len(frames), interval=10)

        court = plt.imread("../match_data/court.png")
        plt.imshow(court, zorder=0, extent=[X_MIN, X_MAX,
                                            Y_MAX, Y_MIN])
        plt.show()


if __name__ == '__main__':
    match = MatchDataParser.MatchParser("../match_data/MIA-DAL.json")
    play_by_play = PlayByPlayParser.EventParser(base_path + match.data['gameid'] + '.csv')
    event = match.get_event()
    a = EventAnimation(event, play_by_play)
