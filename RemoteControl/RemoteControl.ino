#define PIN_X A1
#define PIN_Y A0
#define PIN_Z A6

void setup()
{
  // Mega
  Serial.begin(9600); // comm. with PC
  Serial1.begin(9600); // comm. with xbee
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

  Serial1.write(0xFF);
  Serial1.write(0xFF);
  Serial1.write((byte *)&x, sizeof(x));
  Serial1.write((byte *)&y, sizeof(y));
  Serial1.write((byte *)&z, sizeof(z));
  Serial1.write(0xFF);
  Serial1.write(0xFF);
  
  if(Serial1.available())
  {
    //Serial.println(Serial1.readString());
  }
  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(",");
  Serial.println(z);
  delay(20);

}

