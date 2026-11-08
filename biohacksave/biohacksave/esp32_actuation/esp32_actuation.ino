#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <ESP32Servo.h> 

#define SERVICE_UUID        "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

const int SERVO_PIN = 15; 

BLEServer* pServer = NULL;
Servo myServo;            

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      Serial.println("Laptop connected!");
    };

    void onDisconnect(BLEServer* pServer) {
      Serial.println("Laptop disconnected. Restarting Bluetooth advertising...");
      delay(500); 
      pServer->getAdvertising()->start();
      Serial.println("Advertising restarted! Ready for next test.");
    }
};

class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      String value = pCharacteristic->getValue();
      
      if (value.length() > 0) {
        int blinkCount = (int)value[0]; 
        
        Serial.print("Blink Data Received from Python: ");
        Serial.println(blinkCount);
        
        if (blinkCount == 2) {
          Serial.println("Blink Count is 2! Moving Servo to 180 degrees...");
          myServo.write(180);
          delay(1000);
        }
        else if (blinkCount == 3) {
          Serial.println("Blink Count is 3! Moving Servo to 0 degrees...");
          myServo.write(0);
          delay(1000); 
        } 
        // FIX: Absolutely nothing happens to the servo here if the count is not 2 or 3
        else {
          Serial.println("Value is not 2 or 3. Ignoring transmission completely.");
        }
      }
    }
};

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  
  myServo.setPeriodHertz(50);           
  myServo.attach(SERVO_PIN, 500, 2400); 
  myServo.write(0); // Initialize at base position

  BLEDevice::init("ESP32_Blink_Receiver"); 
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  BLECharacteristic *pCharacteristic = pService->createCharacteristic(
                                         CHARACTERISTIC_UUID,
                                         BLECharacteristic::PROPERTY_WRITE
                                       );
  pCharacteristic->setCallbacks(new MyCallbacks()); 
  pService->start();
  
  pServer->getAdvertising()->start();
  Serial.println("ESP32 ready! Awaiting BLE transmission...");
}

void loop() {
  // Handled by event callbacks
}
