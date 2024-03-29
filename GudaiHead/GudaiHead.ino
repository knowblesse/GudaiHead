#include <SoftwareSerial.h>

#define FL_P 6 // Front Left PWM
#define FL_D 13 // Front Left Direction
#define FL_F true // Front Left front direction

#define FR_P 5
#define FR_D 12
#define FR_F false 

#define BL_P 11
#define BL_D 7
#define BL_F false

#define BR_P 10
#define BR_D 8
#define BR_F true

#define X_default 530
#define Y_default 515
#define Z_default 512

#define speed_limit 0.1d
#define speed_max 1.4142d

SoftwareSerial mySerial (3, 2); //rx2, tx3

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
    analogWrite(motor1P, 200);
    analogWrite(motor2P, 200);
  }
  else if (speed < -speed_limit)
  {
    digitalWrite(motor1D, !motor1F);
    digitalWrite(motor2D, !motor2F);
    analogWrite(motor1P, 200);
    analogWrite(motor2P, 200);
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
double x_, y_, z_;

void loop() 
{
  uint8_t temp;
  uint8_t buffer[8];
  if(mySerial.available())
  {
    
    temp = mySerial.read(); // read first two byte and confirm 0xFF
    Serial.print("*");
    Serial.print(temp, HEX);
    Serial.print(" | ");
    Serial.println(temp);

    if(temp == 0xFF)
    {
      while(!mySerial.available()){}
      temp = mySerial.read();
      if(temp == 0xFE)
      {

        mySerial.readBytes(buffer, 8);
        
        if(buffer[6] == 0xFD & buffer[7] == 0xFB) // read last two byte and confirm 0xFF
        {
          // intact signal received.
          lastSerialTime = millis();

          // parse values
          x = buffer[0] | buffer[1]<<8;
          y = buffer[2] | buffer[3]<<8;
          z = buffer[4] | buffer[5]<<8;

          Serial.print(x);
          Serial.print(',');
          Serial.print(y);
          Serial.print(',');
          Serial.print(z);
          Serial.println("");

          // transform z to -1 ~ +1
          if (z < Z_default) z_ = -(double)(Z_default - z) / (double)Z_default;
          else if (z > Z_default) z_ = (double)(z - Z_default) / (double)(1023 - Z_default);
          else z_ = 0;

          // First check the rotation
          if (abs(z_) > speed_limit)
          {
            // rotate
            setTwoMotorState(z_, FR_P, FR_D, FR_F, BR_P, BR_D, BR_F);
            setTwoMotorState(-z_, FL_P, FL_D, FL_F, BL_P, BL_D, BL_F);
            return;
          }

          // If no rotation is signaled, go movement
          //change x, y range to -1 ~ +1
          if (x < X_default) x_ = -(double)(X_default - x) / (double)X_default;
          else if (x > X_default) x_ = (double)(x - X_default) / (double)(1023 - X_default);
          else x_ = 0;

          if (y < Y_default) y_ = -(double)(Y_default - y) / (double)Y_default;
          else if (y > Y_default) y_ = (double)(y - Y_default) / (double)(1023 - Y_default);
          else y_ = 0;

          // Rotate dimension
          slash_motor = 0.7071 * (-x_ + y_) / speed_max;
          backslash_motor = 0.7071 * (x_ + y_) / speed_max;

          // Change motor state
          setTwoMotorState(slash_motor, FR_P, FR_D, FR_F, BL_P, BL_D, BL_F);
          setTwoMotorState(backslash_motor, FL_P, FL_D, FL_F, BR_P, BR_D, BR_F);

        }
      }
    }
  }
  if (lastSerialTime + timeout < millis())
  {
      setSpeed(0);

  } 
}
