import random

class Mahjong:
    def __init__(self):
        self.players = [Player("Player1"), Player("Player2"), Player("Player3"), Player("Player4")]
        self.round = 1
        self.wind_list = ["东", "南", "西", "北"]
        self.current_wind = self.wind_list[0]
        self.tiles_list = [i for i in range(1, 126)]
        self.win_flag = False
        self.winner = None
        self.discarded_tile = None

    def run(self):
        self.set_up_wind()
        while not self.win_flag:
            if self.round == 1:
                self.set_up_tiles()
            self.round_start()
            while True:
                for player in self.players:
                    if self.win_flag:
                        break
                    player.print_tiles()
                    self.discarded_tile = player.discard_tile()
                    self.check_for_win(player, self.discarded_tile)
                    self.current_player = player
                    
                if self.win_flag:
                    break
            
            self.round += 1

        print("玩家 " + self.winner.name + " 获得胜利！")

    def set_up_wind(self):
        wind_index = self.round % 4
        self.current_wind = self.wind_list[wind_index]

    def set_up_tiles(self):
        random.shuffle(self.tiles_list)
        for i in range(len(self.players)):
            start = i * 13
            end = start + 13
            self.players[i].draw_tiles(self.tiles_list[start:end])
        self.tiles_in_wall = self.tiles_list[52:]

    def round_start(self):
        for player in self.players:
            player.start_round(self.current_wind, self.tiles_in_wall)

    def check_for_win(self, player, discarded_tile):
        for other_player in self.players:
            if player == other_player:
                continue
            win_flag, winning_tiles = player.try_win(other_player, discarded_tile)
            if win_flag:
                self.win_flag = True
                self.winner = player
                break

class Player:
    def __init__(self, name):
        self.name = name
        self.tiles = []
        self.triplets = []
        self.singles = []

    def print_tiles(self):
        print("你的当前手牌：")
        temp_tiles = self.tiles[:]
        temp_tiles.sort()
        for tile in temp_tiles:
            print(tile, end=" ")
        print("")

    def start_round(self, current_wind, tiles_in_wall):
        self.current_wind = current_wind
        self.tiles_in_wall = tiles_in_wall

    def draw_tiles(self, tiles):
        self.tiles += tiles
        self.tiles.sort()

    def discard_tile(self):
        self.print_tiles()
        discarded_tile = input("请打出一张牌：")
        self.tiles.remove(int(discarded_tile))
        return int(discarded_tile)

    def try_win(self, player, discarded_tile):
        temp_tiles = self.tiles[:]
        temp_tiles.append(discarded_tile)
        temp_tiles.sort()

        num_tiles = len(temp_tiles)
        for i in range(num_tiles - 2):
            # 如果连续的三张牌相等，则为刻
            if temp_tiles[i] == temp_tiles[i+1] and temp_tiles[i+1] == temp_tiles[i+2]:
                temp_tiles[i], temp_tiles[i+1], temp_tiles[i+2] = -1, -1, -1
                self.triples.append([temp_tiles[i], temp_tiles[i+1], temp_tiles[i+2]])
        
        # 将temp_tiles中剩余的牌看作单牌
        temp_tiles = [tile for tile in temp_tiles if tile != -1]
        num_singles = len(temp_tiles)
        
        # 雀头可以是任意两张牌
        for i in range(num_singles):
            for j in range(i + 1, num_singles):
                if temp_tiles[i] == temp_tiles[j]:
                    self.singles.append([temp_tiles[i], temp_tiles[j]])
                    temp_tiles[i], temp_tiles[j] = -1, -1
                    win_flag = self.check_tiles(temp_tiles, 0)
                    if win_flag:
                        return True, temp_tiles
                    self.singles.pop()
                    temp_tiles[i], temp_tiles[j] = self.singles[-1]
        
        # 如果没有找到胡牌的组合，则返回False
        return False, None

    def check_tiles(self, tiles, j):
        if len(tiles) == 0:
            return True

        tile = tiles[j]
        if j < len(tiles) - 2 and tiles[j] == tiles[j+1] and tiles[j] == tiles[j+2]:
            # 刻子
            return self.check_tiles(tiles[:j] + tiles[j+3:], j)
        elif tile + 1 in tiles and tile + 2 in tiles:
            # 顺子
            i = tiles.index(tile + 1)
            k = tiles.index(tile + 2)
            return self.check_tiles(tiles[:j] + tiles[j+1:i] + tiles[i+1:k] + tiles[k+1:], min(j, i, k))
        else:
            # 既不是刻子，也不是顺子，则必须继续检测下一张牌
            return self.check_tiles(tiles, j+1)

if __name__ == "__main__":
    game = Mahjong()
    game.run()
