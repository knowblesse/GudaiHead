#include <SoftwareSerial.h>

#define PIN_X A1
#define PIN_Y A0
#define PIN_Z A2

SoftwareSerial btModule(5, 6);

void setup()
{
  Serial.begin(9600); // comm. with PC
  btModule.begin(9600); // comm. with xbee
  Serial.println("Start Remote Control");

}

int x;
int y;
int z;

double slash_motor;
double backslash_motor;

void loop()
{
  x = analogRead(PIN_X);
  y = analogRead(PIN_Y);
  z = analogRead(PIN_Z);

  btModule.write(0xFF);
  btModule.write(0xFE);
  btModule.write((byte *)&x, sizeof(x));
  btModule.write((byte *)&y, sizeof(y));
  btModule.write((byte *)&z, sizeof(z));
  btModule.write(0xFD);
  btModule.write(0xFB);
  
  // Serial.print(x);
  // Serial.print(",");
  // Serial.print(y);
  // Serial.print(",");
  // Serial.println(z);
  delay(60);
}

