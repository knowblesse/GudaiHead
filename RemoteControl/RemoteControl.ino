#define PIN_X A1
#define PIN_Y A0

#define X_default 530
#define Y_default 515

void setup()
{
  // Mega
  Serial.begin(9600);
  Serial1.begin(9600);
}

int x;
int y;

double slash_motor;
double backslash_motor;

uint8_t buffer[7];

void loop()
{
  x = analogRead(PIN_X);
  y = analogRead(PIN_Y);

  x = 128;
  y = 15;
  int z = 19;

  Serial1.write(0xFF);
  Serial1.write(0xFF);
  Serial1.write((byte *)&x, sizeof(x));
  Serial1.write((byte *)&y, sizeof(y));
  Serial1.write((byte *)&z, sizeof(z));
  
  //Serial1.println(x + 0.0001*(double)y,4);
  //Serial.println(x + 0.0001*(double)y,4);
  if(Serial1.available())
  {
    //Serial.println(Serial1.readString());
  }
}
