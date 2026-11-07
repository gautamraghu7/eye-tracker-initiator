const int PINCH_SIGNAL_PIN = 25;

// Safety timeout
const unsigned long WATCHDOG_TIMEOUT = 500;

unsigned long lastCommandTime = 0;

bool isPinching = false;


void setPinch(bool pinch) {

  isPinching = pinch;

  // HIGH = pinch signal
  // LOW = release signal
  digitalWrite(
    PINCH_SIGNAL_PIN,
    pinch ? HIGH : LOW
  );

  if (pinch) {
    Serial.println("MECHANISM: PINCH");
  } else {
    Serial.println("MECHANISM: RELEASE");
  }
}


void setup() {

  Serial.begin(115200);

  pinMode(PINCH_SIGNAL_PIN, OUTPUT);

  // Safe startup state
  setPinch(false);

  lastCommandTime = millis();

}


void loop() {

  // Receive commands from Python
  if (Serial.available()) {

    String command = Serial.readStringUntil('\n');

    command.trim();

    if (command == "PINCH") {

      setPinch(true);
      lastCommandTime = millis();

    }

    else if (command == "RELEASE") {

      setPinch(false);
      lastCommandTime = millis();

    }

  }

  // Watchdog: release if communication stops
  if (
    isPinching &&
    millis() - lastCommandTime > WATCHDOG_TIMEOUT
  ) {

    setPinch(false);

    Serial.println("WATCHDOG: EMERGENCY RELEASE");

  }

}

