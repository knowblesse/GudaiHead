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
void sendATCommand()
{
  for(int i = 0; i < 3; i++)
  {
    unsigned long startTime = millis();
    unsigned long timeout = 6000;
    unsigned long temp;

    Serial1.flush();
    delay(1000); // Go slience for one sec
    
    Serial.println("Entering the AT mode");
    Serial1.write("+++"); // Entering the AT mode
    temp = millis();
    
    while (!Serial1.available() && (millis() - startTime) < timeout){} // wait until zigbee issue "OK\r"
    if (!Serial1.available())
    {
      Serial.println("timeout");
      continue;
    }
    else
    {
      Serial.println(millis() - temp);
      Serial.println(Serial1.readStringUntil('\r'));
    }
    
    // Now Send AT commands
    while(Serial.available())
    {
      Serial1.write(Serial.read()); // write AT command
    } 
    Serial1.write('\r'); // write carrage return (set "No Line Ending" for the Serial monitor)
    
    // Wait Response
    while (!Serial1.available() && (millis() - startTime) < timeout){} // wait until zigbee issue "OK\r"
    if (!Serial1.available())
    {
      Serial.println("timeout");
      continue;
    }
    else
    {
      while(Serial1.available())
      {
        Serial.print(Serial1.readString());
      }
    }
    Serial.println(' ');
    Serial.println("response received");
    Serial1.write("ATCN\r");
    return;
  }
}
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
  
  if(Serial.available())
  {
    sendATCommand();
  }
  // Serial.print(x);
  // Serial.print(",");
  // Serial.print(y);
  // Serial.print(",");
  // Serial.println(z);
  delay(20);
}

