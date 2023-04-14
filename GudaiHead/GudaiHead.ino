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
}

double readValue;

double slash_motor;
double backslash_motor;

unsigned long lastSerialTime;
unsigned long timeout = 1000;

int x, y, z;
uint8_t temp;

uint8_t readSerial()
{
  if(mySerial.available())
  {
    return mySerial.read();
  }
}

void loop() {
  temp = readSerial();
  Serial.println(temp, HEX);
  if(temp == 0xFF)
  {
    temp = readSerial();
    Serial.println(temp, HEX);
    if(temp == 0xFF)
    {
      mySerial.readBytes((byte *)&x, sizeof(x));
      mySerial.readBytes((byte *)&y, sizeof(y));
      mySerial.readBytes((byte *)&z, sizeof(z));

      Serial.print(x);
      Serial.print(',');
      Serial.print(y);
      Serial.print(',');
      Serial.println(z);

      lastSerialTime = millis();

      // change range to -1 ~ +1
      if (x < X_default) x = -(X_default - x) / X_default;
      else if (x > X_default) x = (x - X_default) / (1023 - X_default);
      else x = 0;

      if (y < Y_default) y = -(Y_default - y) / Y_default;
      else if (y > Y_default) y = (y - Y_default) / (1023 - Y_default);
      else y = 0;

      // Rotate
      slash_motor = 0.7071 * (-x + y);
      backslash_motor = 0.7071 * (x + y);

      setTwoMotorState(slash_motor, FR_P, FR_D, FR_F, BL_P, BL_D, BL_F);
      setTwoMotorState(backslash_motor, FL_P, FL_D, FL_F, BR_P, BR_D, BR_F);
    };
  }

  if (lastSerialTime + timeout < millis())
  {
      setSpeed(0);
  } 
}