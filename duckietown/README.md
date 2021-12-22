# Duckietown

Слева -- исходное изображение.

По центру -- проекция "вид сверху". Зеленый прямоугольник обводит утку, зеленая линия показывает желтую центральную линию. Красные линии показывает белые ограничительные линии. Синяя штука снизу - робот. Если чего-то из этого нет, значит в данном кадре оно не определилось. 

Справа -- скорректированная проекция по определенному углу и положению робота. Если кадр дергается, значит в нем позиция или угол не определились.

Ноутбук, с помощью которого было получено видео, в репозитории.

Логи запуска можно найти (здесь)[https://github.com/OSLL/aido-auto-feedback/tree/ba8f906720343884fde51239f33704515a854126c913509a059eb883/dont_crush_duckie].

## Описание алгоритма робота

* До утки робот просто едет прямо
* Затем совершается объезд по скрипту. Да, говорили не делать скрипты, но я не понимаю как, ведь во время объезда утку не видно. Можно было пытаться интегрировать скорость и считать координаты, можно было постоянно останавливаться, поворачиваться на утку, и понимать, где она. Но мне все это показалось странным.
* **НО** после скрипта, робот 20 итераций едет вперед, удерживая скорость пропорциональным регулятором, ориентируясь на свою позицию и угол.