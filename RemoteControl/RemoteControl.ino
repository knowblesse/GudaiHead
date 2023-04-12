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

double x;
double y;

double slash_motor;
double backslash_motor;

void loop()
{
  x = analogRead(PIN_X);
  y = analogRead(PIN_Y);

  Serial1.println(x + 0.0001*(double)y,4);
  Serial.println(x + 0.0001*(double)y,4);
  if(Serial1.available())
  {
    //Serial.println(Serial1.readString());
  }
  delay(20);
}
