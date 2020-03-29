/* Use a variable called byteRead to temporarily store
   the data coming from the computer */
#include <Arduino.h>
#include <Servo.h>

// Use Serial monitor to input speed 
int poser = 0; 
int val; 
int speed = 0;
String incomingByte;

// motor one variables
int enB = 5;
int in3 = 7;
int in4 = 6;

int delx = 250; // flashed ==> 250
int dely = 1000; // flashed ==> 250
int pos;

void setup() {
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);

  pinMode(enB, OUTPUT);  // Initialising all
  pinMode(in3, OUTPUT);   // motor 
  pinMode(in4, OUTPUT);   // pins
}

void ReactorPerform() {
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
  
  if (Serial.available())
  {
    incomingByte = Serial.readStringUntil('\n');   
    speed = incomingByte.toInt();
    Serial.print("Running at speed ==> ");
    Serial.println(speed);
    analogWrite(enB, speed);
  }
}

void loop() {
  int sensorValue = analogRead(A0);
  // Serial.println("The sensor value is :");
  // Serial.println(sensorValue);
  delay(10);
  ReactorPerform();  // activating the arc reactor 
}
