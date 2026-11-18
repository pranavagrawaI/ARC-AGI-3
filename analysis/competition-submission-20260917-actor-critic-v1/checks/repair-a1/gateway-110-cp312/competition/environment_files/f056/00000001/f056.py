from arcengine import ARCBaseGame,Level,Sprite
class F056(ARCBaseGame):
    def __init__(self,seed=0):
        self.n=0
        super().__init__(game_id='f056',levels=[Level(sprites=[Sprite(pixels=[[3]],name="piece")],grid_size=(8,8)),Level(sprites=[Sprite(pixels=[[4]],name="piece")],grid_size=(8,8))],available_actions=[1],seed=seed)
    def on_set_level(self,level): self.n=0
    def step(self):
        self.n+=1
        self.current_level.get_sprites()[0].set_position(self.n, 0)
        if self.n==2:self.next_level()
        self.complete_action()
