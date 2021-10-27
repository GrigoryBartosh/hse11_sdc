## Запуск

В **первом** терминале
```
roscore
```

Во **втором** терминале
```
rosrun turtlesim turtlesim_node
```

В **третьем** терминале
```
rosservice call /spawn "x: 1.0
y: 1.0
theta: 0.0
name: 'raphael'"
```

Затем
```
cd hse11_sdc/turtles_ws
catkin_make
source devel/setup.bash
rosrun turtles turtle.py
```

В **четвертом** терминале
```
rosrun turtlesim turtle_teleop_key
```
