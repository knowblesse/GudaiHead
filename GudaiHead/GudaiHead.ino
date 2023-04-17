#include <SoftwareSerial.h>

#define FL_P 6 // Front Left PWM
#define FL_D 13 // Front Left Direction
#define FL_F false // Front Left front direction

#define FR_P 5
#define FR_D 12
#define FR_F false 

#define BL_P 11
#define BL_D 7
#define BL_F true

#define BR_P 10
#define BR_D 8
#define BR_F true

#define X_default 530
#define Y_default 515
#define Z_default 512

#define speed_limit 0.1
#define speed_max 1.4142

SoftwareSerial mySerial (2, 3); //rx2, tx3

void goUp()
{
  digitalWrite(FL_D, FL_F);
  digitalWrite(FR_D, FR_F);
  digitalWrite(BL_D, BL_F);
  digitalWrite(BR_D, BR_F);
}


void goDown()
{
  digitalWrite(FL_D, !FL_F);
  digitalWrite(FR_D, !FR_F);
  digitalWrite(BL_D, !BL_F);
  digitalWrite(BR_D, !BR_F);
}

void goRight()
{
  digitalWrite(FL_D, FL_F);
  digitalWrite(FR_D, !FR_F);
  digitalWrite(BL_D, BL_F);
  digitalWrite(BR_D, !BR_F);
}

void goLeft()
{
  digitalWrite(FL_D, !FL_F);
  digitalWrite(FR_D, FR_F);
  digitalWrite(BL_D, !BL_F);
  digitalWrite(BR_D, BR_F);
}

void setSpeed(byte speed)
{
  analogWrite(FL_P, (int)speed);
  analogWrite(FR_P, (int)speed);
  analogWrite(BL_P, (int)speed);
  analogWrite(BR_P, (int)speed);
}

void setTwoMotorState(double speed, int motor1P, int motor1D, bool motor1F, int motor2P, int motor2D, bool motor2F)
{
  if (speed > speed_limit)
  {
    digitalWrite(motor1D, motor1F);
    digitalWrite(motor2D, motor2F);
    analogWrite(motor1P, round(speed / speed_max * 255));
    analogWrite(motor2P, round(speed / speed_max * 255));
  }
  else if (speed < -speed_limit)
  {
    digitalWrite(motor1D, !motor1F);
    digitalWrite(motor2D, !motor2F);
    analogWrite(motor1P, round(speed / (-speed_max) * 255));
    analogWrite(motor2P, round(speed / (-speed_max) * 255));
  }
  else
  {
    analogWrite(motor1P, 0);
    analogWrite(motor2P, 0);
  }
}

unsigned long startMovingTime;
void setup() {

  Serial.begin(9600);
  mySerial.begin(9600);

  pinMode(FL_P, OUTPUT);
  pinMode(FL_D, OUTPUT);

  pinMode(FR_P, OUTPUT);
  pinMode(FR_D, OUTPUT);

  pinMode(BL_P, OUTPUT);
  pinMode(BL_D, OUTPUT);

  pinMode(BR_P, OUTPUT);
  pinMode(BR_D, OUTPUT);

  delay(8000);
  goUp();
  setSpeed(180);
  startMovingTime = millis();

}

double readValue;

double slash_motor;
double backslash_motor;

unsigned long lastSerialTime;
unsigned long timeout = 1000;

int x, y, z;
int* remoteValue[3];

bool movemove = true; // for habituation

bool isMovingUp = false;
unsigned long randomTime = random(10000);
void loop() 
{
  randomTime = random(3000,5000);
  delay(randomTime);
  
  randomTime = random(3000,5000);
  setSpeed(0);
  delay(randomTime);
  goUp();
  setSpeed(180);
  
}