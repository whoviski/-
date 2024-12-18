import time
import threading
from queue import Queue
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


class TrafficLight:
    def __init__(self, id, is_pedestrian=False):
        self.id = id
        self.is_pedestrian = is_pedestrian
        self.state = 'red'
        self.car_queue = 0
        self.pedestrian_queue = 0
        self.event_queue = Queue()
        self.lock = threading.Lock()
        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(2,), kernel_initializer='he_normal'),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_initializer='he_normal'),
            Dense(16, activation='relu', kernel_initializer='he_normal'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def change_state(self, new_state):
        with self.lock:
            self.state = new_state
            print(f"Traffic Light {self.id} changed to {self.state}")

    def process_event(self):
        while True:
            event = self.event_queue.get()
            if event is None:  # Stop signal
                break
            self.car_queue += event['cars']
            self.pedestrian_queue += event['pedestrians']
            print(f"Traffic Light {self.id} received event: {event}")
            self.adaptive_control()

    def adaptive_control(self):
        with self.lock:
            if not self.is_pedestrian and self.car_queue > 5:  
                self.change_state('green')
                optimal_time = self.predict_optimal_time(self.car_queue, self.pedestrian_queue)
                time.sleep(optimal_time)
                self.car_queue = max(0, self.car_queue - 5)
                self.change_state('yellow')
                time.sleep(2)
                self.change_state('red')

            elif self.is_pedestrian and self.pedestrian_queue > 3:
                self.change_state('green')
                optimal_time = self.predict_optimal_time(self.car_queue, self.pedestrian_queue)
                time.sleep(optimal_time)
                self.pedestrian_queue = max(0, self.pedestrian_queue - 3)
                self.change_state('red')

            else:
                if self.state != 'red':
                    self.change_state('red')

    def predict_optimal_time(self, car_count, pedestrian_count):
        input_data = np.array([[car_count, pedestrian_count]])
        prediction = self.model.predict(input_data)
        return max(int(prediction[0][0]), 5)  
    def send_event(self, traffic_light_id, cars=0, pedestrians=0):
        event = {'cars': cars, 'pedestrians': pedestrians}
        print(f"Traffic Light {self.id} sending event to {traffic_light_id}: {event}")
        traffic_lights[traffic_light_id].event_queue.put(event)


car_traffic_lights = {i: TrafficLight(i) for i in range(4)}  # 4 car traffic lights
pedestrian_traffic_lights = {i: TrafficLight(i + 4, is_pedestrian=True) for i in
                             range(8)}

for light in car_traffic_lights.values():
    threading.Thread(target=light.process_event).start()

for light in pedestrian_traffic_lights.values():
    threading.Thread(target=light.process_event).start()

time.sleep(1)

for _ in range(10):
    light_id = random.choice(list(car_traffic_lights.keys()))
    cars = random.randint(0, 10)
    pedestrians = random.randint(0, 5)
    car_traffic_lights[light_id].send_event(light_id, cars=cars)

    light_id = random.choice(list(pedestrian_traffic_lights.keys()))
    pedestrians = random.randint(0, 5)
    pedestrian_traffic_lights[light_id].send_event(light_id + 4, pedestrians=pedestrians)

    time.sleep(random.uniform(30, 60)) 

time.sleep(20)
for light in car_traffic_lights.values():
    light.event_queue.put(None)

for light in pedestrian_traffic_lights.values():
    light.event_queue.put(None)

print("Simulation stopped")
