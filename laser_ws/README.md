## Запуск

В **первом** терминале
```
roscore
```

Во **втором** терминале
```
catkin_make
source devel/setup.bash
rosrun turtles turtle.py
```

В **третьем** терминале
```
rosbag play 2011-01-25-06-29-26.bag
```

В **четвертом** терминале
```
rviz -d laser.rviz
```
