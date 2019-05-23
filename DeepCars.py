import pygame, random, os, time, sys
import  numpy as np
from pygame.locals import *

# =======================================================================================================================
# -------------------------------------------Global Parameter values----------------------------------------------------
# =======================================================================================================================

DefaultTextColor = (255, 97, 3)
BackgroundColor = (255, 255, 255)
FPS = 100

Horiz_Move_Rate = 4
AddNewCarRate = 1
CarsMinSpeed = 2  # Min. realative speed for the other cars wrt to the PlayerSpeed
CarsMaxSpeed = 2  # Max. realative speed for the other cars wrt to the PlayerSpeed
PlayerSpeed = 8  # Player's speed wrt to side walls (ground speed)

CarWidth = 50
CarHeight = 100
SpaceWidth = 5  # Width of space between objects, e.g. btw car and line/wall
LineWidth = 5  # Width of the lines in between the lanes (dashed lines)
LineHeight = 25
WallWidth = 50  # Width of the walls on the left and right sides
NoOfLanes = 5
MaxCarsInLane = 5  # Maximum number of cars vertically in one lane (for window size setting)

ActionList = ['Left', 'Stay', 'Right']

# Define window's dimensions wrt objects dimensions
WindowWidth = (CarWidth + 2 * SpaceWidth) * NoOfLanes + LineWidth * (NoOfLanes - 1) + 2 * WallWidth
WindowHeight = CarHeight * MaxCarsInLane + 2*SpaceWidth + (MaxCarsInLane-1)*LineWidth

# Calculate the x coordinate of top-right pixel of cars for staying in all lanes
LaneXCoorVec = []
for i in range(NoOfLanes):
    tempX = WallWidth + SpaceWidth + i * (CarWidth + SpaceWidth + LineWidth + SpaceWidth)
    LaneXCoorVec.append(tempX)

# No of pixels in between the vertical grids
NoOfHorGridPixels = LaneXCoorVec[1] - LaneXCoorVec[0]

# Calculate the y coordinate of top-right pixel of cars for staying in all grid rectangles
LaneYCoorVec = []
for i in range(MaxCarsInLane):
    tempY = SpaceWidth + i*(CarHeight + SpaceWidth + LineWidth + SpaceWidth)
    LaneYCoorVec.append(tempY)

# No of pixels in between the vertical grids
NoOfVerGridPixels = LaneYCoorVec[1] - LaneYCoorVec[0]

# =======================================================================================================================
# -------------------------------------------Grid World Class-----------------------------------------------------------
# =======================================================================================================================

class GridWorld:

    def __init__(self):
        # Initializations
        self.MainClock = 0
        self.WindowSurface = 0
        self.font = 0
        self.PlayerImage = 0
        self.CarsImageVec = []
        self.LineImage = 0
        self.HorizLineImage = 0
        self.LeftWall = 0
        self.RightWall = 0
        self.LineRecSamples = []
        self.HorizLineRecSamples = []

        self.CarAddCounter = AddNewCarRate
        self.PassedCarsCount = 1  # No. of cars that the agent has passed (start from 1 to avoid deving to 0 in SuccessRate)
        self.HitCarsCount = 0  # No. of cars that are hit by player
        self.OtherCarsVec = []
        self.PlayerLane = int(round((NoOfLanes - 1) / 2))
        self.PlayerRect = 0

    def Terminate(self):
        pygame.quit()
        print("The game is terminated")
        # sys.exit()

    def ObservationSpace(self):
        return NoOfLanes+1          # State vector size

    def ActionSpace(self):
        return 3                    # Action vector size: [Left Stay Right]

    def Reset(self):                # Get initial state
        StateVec = []
        for i in range(0, NoOfLanes + 1):
            StateVec.append(MaxCarsInLane - 2)  # [Player Lane Number  ,   Distance to the car in front in lane (i) ]
        StateVec[0] = self.PlayerLane
        return StateVec

    def WaitForPlayerToPressKey(self):
        print("Press a key to continue or Esc to terminate")
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.Terminate()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:  # escape quits
                        self.Terminate()
                    return


    def DrawText(self,text, font, TextColor, surface, x, y):
        textobj = font.render(text, 1, TextColor)
        textrect = textobj.get_rect()
        textrect.topleft = (x, y)
        surface.blit(textobj, textrect)

    def PlayerHasHitBaddie(self,PlayerRect, baddies):
        for b in baddies:
            if PlayerRect.colliderect(b['rec']):
                return True
        return False

    def PygameInitialize(self):
        # set up pygame, the window, and the mouse cursor
        pygame.init()
        self.MainClock = pygame.time.Clock()
        self.WindowSurface = pygame.display.set_mode((WindowWidth, WindowHeight))
        pygame.display.set_caption('Deep Cars Grid World (ITUarc)')
        pygame.mouse.set_visible(False)
        self.font = pygame.font.SysFont(None, 30)

        # images
        print(os.getcwd())
        self.PlayerImage = pygame.image.load('image/MyCar')
        self.PlayerImage = pygame.transform.scale(self.PlayerImage, (CarWidth, CarHeight))

        Car1Image = pygame.image.load('image/Car1')
        Car1Image = pygame.transform.scale(Car1Image, (CarWidth, CarHeight))
        Car2Image = pygame.image.load('image/Car2')
        Car2Image = pygame.transform.scale(Car2Image, (CarWidth, CarHeight))
        Car3Image = pygame.image.load('image/Car3')
        Car3Image = pygame.transform.scale(Car3Image, (CarWidth, CarHeight))
        Car4Image = pygame.image.load('image/Car4')
        Car4Image = pygame.transform.scale(Car4Image, (CarWidth, CarHeight))
        Car5Image = pygame.image.load('image/Car5')
        Car5Image = pygame.transform.scale(Car5Image, (CarWidth, CarHeight))

        self.CarsImageVec = [Car1Image, Car2Image, Car3Image, Car4Image, Car5Image]

        LeftWallImage = pygame.image.load('image/left')
        RightWallImage = pygame.image.load('image/right')

        self.LineImage = pygame.image.load('image/black_line')
        self.LineImage = pygame.transform.scale(self.LineImage, (LineWidth, WindowHeight))

        self.HorizLineImage = pygame.image.load('image/Horizontal_Dashes')
        self.HorizLineImage = pygame.transform.scale(self.HorizLineImage, (WindowWidth, LineWidth))

        # Define walls
        self.LeftWall = {'rec': pygame.Rect(0, -2 * WindowHeight, WallWidth, 3 * WindowHeight),
                    'surface': pygame.transform.scale(LeftWallImage, (WallWidth, 3 * WindowHeight))
                    }
        self.RightWall = {'rec': pygame.Rect(WindowWidth - WallWidth, -2 * WindowHeight, WallWidth, 3 * WindowHeight),
                     'surface': pygame.transform.scale(RightWallImage, (WallWidth, 3 * WindowHeight))
                     }

        # Define line rectangles
        for i in range(NoOfLanes - 1):
            LineXCoord = WallWidth + (i) * LineWidth + (i + 1) * (SpaceWidth + CarWidth + SpaceWidth)
            NewLineRec = pygame.Rect(LineXCoord, 0, LineWidth, WindowHeight)
            self.LineRecSamples.append(NewLineRec)

        for i in range(MaxCarsInLane - 1):
            LineYCoord = LaneYCoorVec[i + 1]
            NewLineRec = pygame.Rect(0, LineYCoord, WindowWidth, LineWidth)
            self.HorizLineRecSamples.append(NewLineRec)

        self.PlayerRect = self.PlayerImage.get_rect()
        self.PlayerRect.topleft = (LaneXCoorVec[self.PlayerLane], LaneYCoorVec[MaxCarsInLane - 2])

        print('The game has initiated')

    def update(self,ActionIndex,TrainingFlag):

        Action = ActionList[ActionIndex]        # Pick the action from action list

        # ==============================================Define new cars======================================================
        if self.CarAddCounter >= AddNewCarRate:
            self.CarAddCounter = 0
            NewCarLaneNo = random.randint(0, NoOfLanes - 1)
            NewCar = {'rec': pygame.Rect(LaneXCoorVec[NewCarLaneNo], 0 - CarHeight, CarWidth, CarHeight),
                      'speed': NoOfVerGridPixels,
                      'XCoord': NewCarLaneNo,           # x coordinate in grid world
                      'YCoord': MaxCarsInLane - 1,      # y coordinate in grid world
                      'surface': self.CarsImageVec[random.randint(0, len(self.CarsImageVec) - 1)]  # Randomize cars visuals
                      }
            self.OtherCarsVec.append(NewCar)
        self.CarAddCounter += 1

        # =================================================Movements=========================================================
        # Side walls
        self.LeftWall['rec'].move_ip(0, NoOfVerGridPixels)
        self.RightWall['rec'].move_ip(0, NoOfVerGridPixels)

        # Move the player left
        if Action is 'Left' and self.PlayerLane is not 0:
            self.PlayerRect.move_ip(-1 * NoOfHorGridPixels, 0)
            self.PlayerLane -= 1
            # print('Player car lane number has chnaged to ', update.PlayerLane + 1)

        # Move the player right
        if Action is 'Right' and self.PlayerLane is not NoOfLanes - 1:
            self.PlayerRect.move_ip(+1 * NoOfHorGridPixels, 0)
            self.PlayerLane += 1
            # print('Player car lane number has chnaged to ', update.PlayerLane + 1)

        # Move other cars backward
        for Car in self.OtherCarsVec:
            Car['rec'].move_ip(0, +1 *Car['speed'])
            Car['YCoord'] -= 1

        # ================================Remove the other cars that pass the game window====================================
        for Car in self.OtherCarsVec:
            #if Car['rec'].top > WindowHeight + SpaceWidth or Car['rec'].top + CarHeight < - SpaceWidth:
            if Car['YCoord'] < 0:
                self.OtherCarsVec.remove(Car)
                self.PassedCarsCount += 1

        # ==================================================================================================================
        # -----------------------------------------------State and reward---------------------------------------------------
        # ==================================================================================================================
        StateVec = []
        for i in range(0,NoOfLanes+1):
            StateVec.append(MaxCarsInLane - 2)      # [Player Lane Number  ,   Distance to the car in front in lane (i) ]
        StateVec[0] = self.PlayerLane

        for Car in self.OtherCarsVec:
            if Car['YCoord'] < StateVec[Car['XCoord']+1]:
            # Car['YCoord'] < StateVec[Car['XCoord']+1]:  ===>  For more than one car in the same lane, select the closer one
                StateVec[Car['XCoord']+1] = Car['YCoord']    # Number of grid rectangles existing in between (including car rectangle)

        if self.PlayerHasHitBaddie(self.PlayerRect,self.OtherCarsVec):
            Reward = -100
            self.PassedCarsCount -= 1
            self.HitCarsCount += 1
        else:
            Reward = 1

        # =======================================Draw the game world on the window===========================================
        self.WindowSurface.fill(BackgroundColor)

        for i in range(0, len(self.LineRecSamples)):
            self.WindowSurface.blit(self.LineImage, self.LineRecSamples[i])

        for i in range(0, len(self.HorizLineRecSamples)):
            self.WindowSurface.blit(self.HorizLineImage, self.HorizLineRecSamples[i])

        self.WindowSurface.blit(self.PlayerImage, self.PlayerRect)

        for Car in self.OtherCarsVec:
            self.WindowSurface.blit(Car['surface'], Car['rec'])

        self.DrawText('Cars passed: %s' % (self.PassedCarsCount), self.font, DefaultTextColor, self.WindowSurface, WallWidth + SpaceWidth,
                 SpaceWidth)
        self.DrawText('Cars hit: %s' % (self.HitCarsCount), self.font, DefaultTextColor, self.WindowSurface, WallWidth + SpaceWidth,
                 10*SpaceWidth)

        self.WindowSurface.blit(self.LeftWall['surface'], self.LeftWall['rec'])
        self.WindowSurface.blit(self.RightWall['surface'], self.RightWall['rec'])

        # ============Move walls' Y coordinate to -2*WindowHeight when their top left pixel reached to y = 0================
        if self.LeftWall['rec'].topleft[1] >= - WindowHeight / 2:
            self.LeftWall['rec'].topleft = [0, -2 * WindowHeight]
            self.RightWall['rec'].topleft = [WindowWidth - WallWidth, -2 * WindowHeight]

        # ==============================================Environment update==================================================
        SuccessRate = round(100 * self.PassedCarsCount / (self.PassedCarsCount + self.HitCarsCount), 2)
        if TrainingFlag is True:            # Test part
            self.DrawText('Training: %s' % (SuccessRate), self.font, DefaultTextColor, self.WindowSurface, WallWidth + SpaceWidth,
                     20 * SpaceWidth)

        else:
            self.DrawText('Test: %s' % (SuccessRate), self.font, DefaultTextColor, self.WindowSurface, WallWidth + SpaceWidth,
                     20 * SpaceWidth)

        pygame.display.update()
        self.MainClock.tick(FPS)

        # ==================================================================================================================
        # -----------------------------------------------ESC for Terminate--------------------------------------------------
        # ==================================================================================================================
        IsTerminated = False
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:  # escape quits
                    IsTerminated = True
                    self.Terminate()

        return StateVec, Reward, IsTerminated, self.HitCarsCount, self.PassedCarsCount

if __name__ == "__main__":

    from DeepCars import GridWorld as envObj
    env = envObj()
    env.PygameInitialize()
    while True:
        env.update(ActionIndex=1,TrainingFlag=False)
        time.sleep(0.1)
