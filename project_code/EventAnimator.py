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
    def __init__(self, event):
        self.event = event
        self.team_info = self.get_team_details()
        self.event_frames = self.get_event_frames()
        self.show_event()

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
            frame = {SHOTCLOCK: moment[3], COORDINATES: player_position}
            frames.append(frame)
        return frames

    def update_positions(self, i, player_circles, player_annotations):
        frame = self.event_frames[i]
        player_coordinates = frame[COORDINATES]
        for j, circle in enumerate(player_circles):
            circle.center = player_coordinates[j][X_POSITION], player_coordinates[j][Y_POSITION]
            player_annotations[j].set_position(circle.center)
        player_circles[0].radius = player_coordinates[0][Z_POSITION]/7
        return player_circles

    def show_event(self):
        axs = plt.axes(ylim=(Y_MIN, Y_MAX), xlim=(X_MIN, X_MAX))
        axs.axis('off')
        axs.grid(False)
        frames = self.event_frames
        team_info = self.team_info
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

        anim = animation.FuncAnimation(
            figure, self.update_positions,
            fargs=(player_circles, player_annotations),
            frames=len(frames), interval=10)

        court = plt.imread("../match_data/court.png")
        plt.imshow(court, zorder=0, extent=[X_MIN, X_MAX,
                                            Y_MAX, Y_MIN])
        plt.show()


if __name__ == '__main__':
    match = MatchDataParser.MatchParser("../match_data/MIA-DAL.json")
    play_by_play = PlayByPlayParser.EventParser(base_path + match.data['gameid'])
    event = match.get_event()
    a = EventAnimation(event)
