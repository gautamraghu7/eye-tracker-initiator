#include "BluetoothSerial.h"


// ==========================================
// 1. BLUETOOTH SETUP
// ==========================================

BluetoothSerial SerialBT;


// ==========================================
// 2. HARDWARE SETTINGS
// ==========================================

const int PINCH_SIGNAL_PIN = 25;


// ==========================================
// 3. WATCHDOG SETTINGS
// ==========================================

const unsigned long WATCHDOG_TIMEOUT = 500;


// ==========================================
// 4. VARIABLES
// ==========================================

unsigned long lastCommandTime = 0;

bool isPinching = false;

String receivedCommand = "";


// ==========================================
// 5. MECHANISM CONTROL
// ==========================================

void setPinch(bool pinch) {

    // Update mechanism state

    isPinching = pinch;


    // HIGH = GRIP
    // LOW = RELEASE

    digitalWrite(
        PINCH_SIGNAL_PIN,
        pinch ? HIGH : LOW
    );


    // Print only when state changes

    Serial.println(
        pinch
        ? "MECHANISM: GRIP"
        : "MECHANISM: RELEASE"
    );

}


// ==========================================
// 6. PROCESS BLUETOOTH COMMANDS
// ==========================================

void processCommand(String command) {

    command.trim();


    if (command == "PINCH") {

        Serial.println("BLUETOOTH: PINCH RECEIVED");

        setPinch(true);

        lastCommandTime = millis();

    }


    else if (command == "RELEASE") {

        Serial.println("BLUETOOTH: RELEASE RECEIVED");

        setPinch(false);

        lastCommandTime = millis();

    }

}


// ==========================================
// 7. BLUETOOTH SETUP
// ==========================================

void setup() {

    Serial.begin(115200);


    // Configure output pin

    pinMode(
        PINCH_SIGNAL_PIN,
        OUTPUT
    );


    // Start in safe RELEASE state

    digitalWrite(
        PINCH_SIGNAL_PIN,
        LOW
    );

    isPinching = false;

    lastCommandTime = millis();


    // Start Bluetooth Classic

    if (!SerialBT.begin("EYE_CONTROL_ESP32")) {

        Serial.println(
            "ERROR: BLUETOOTH INITIALIZATION FAILED"
        );

        while (true) {

            digitalWrite(
                PINCH_SIGNAL_PIN,
                LOW
            );

            delay(100);

        }

    }


    Serial.println();

    Serial.println("==============================");

    Serial.println("ESP32 BLUETOOTH READY");

    Serial.println("DEVICE: EYE_CONTROL_ESP32");

    Serial.println("WAITING FOR PYTHON CONNECTION");

    Serial.println("==============================");

    Serial.println();

}


// ==========================================
// 8. MAIN LOOP
// ==========================================

void loop() {


    // ======================================
    // RECEIVE BLUETOOTH DATA
    // ======================================

    while (SerialBT.available()) {

        char incoming = SerialBT.read();


        // End of command

        if (incoming == '\n') {

            processCommand(receivedCommand);

            receivedCommand = "";

        }


        // Ignore carriage return

        else if (incoming != '\r') {

            if (receivedCommand.length() < 30) {

                receivedCommand += incoming;

            }

            else {

                // Prevent oversized command strings

                receivedCommand = "";

            }

        }

    }


    // ======================================
    // WATCHDOG SAFETY SYSTEM
    // ======================================

    if (
        isPinching &&
        millis() - lastCommandTime > WATCHDOG_TIMEOUT
    ) {

        Serial.println(
            "WATCHDOG: COMMUNICATION LOST!"
        );

        Serial.println(
            "WATCHDOG: EMERGENCY RELEASE"
        );


        // Automatically release mechanism

        setPinch(false);

    }


    delay(5);

}