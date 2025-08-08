import robomaster
from robomaster import robot
from robomaster import chassis
import time

class SquareMovementSimple:
    def __init__(self):
        # Initialize RoboMaster S1
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type="ap")
        self.ep_chassis = self.ep_robot.chassis
        
        # Movement parameters
        self.square_size = 100  # cm
        self.move_speed = 0.3   # m/s (30 cm/s)
        self.turn_speed = 45    # degrees/s
        self.turn_angle = 90    # degrees
        
        print("RoboMaster S1 initialized successfully!")
        print(f"Square size: {self.square_size} cm")
        print(f"Move speed: {self.move_speed} m/s")
        print(f"Turn speed: {self.turn_speed} deg/s")

    def move_straight(self, distance_cm):
        """Move straight for specified distance"""
        print(f"Moving straight: {distance_cm} cm")
        
        # Calculate movement time (distance in cm, speed in m/s)
        move_time = (distance_cm / 100.0) / self.move_speed
        print(f"Move time: {move_time:.2f} seconds")
        
        # Start moving forward
        self.ep_chassis.drive_speed(x=self.move_speed, y=0, z=0)
        
        # Wait for calculated time
        time.sleep(move_time)
        
        # Stop movement
        self.ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.5)  # Pause for stability

    def turn_right_90(self):
        """Turn 90 degrees clockwise"""
        print("Turning 90 degrees clockwise...")
        
        # Calculate turn time
        turn_time = self.turn_angle / self.turn_speed
        print(f"Turn time: {turn_time:.2f} seconds")
        
        # Start turning clockwise (negative z for clockwise)
        self.ep_chassis.drive_speed(x=0, y=0, z=-self.turn_speed)
        
        # Wait for calculated time
        time.sleep(turn_time)
        
        # Stop turning
        self.ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.5)  # Pause for stability

    def move_square(self, num_squares=1):
        """Execute square movement pattern"""
        print(f"\nStarting square movement pattern")
        print(f"Number of squares: {num_squares}")
        print("=" * 40)
        
        for square in range(num_squares):
            print(f"\n--- Square {square + 1}/{num_squares} ---")
            
            # Move 4 sides of the square
            for side in range(4):
                print(f"\nSide {side + 1}/4:")
                
                # Move straight
                self.move_straight(self.square_size)
                
                # Turn 90 degrees (except after the last side)
                if side < 3:
                    self.turn_right_90()
                else:
                    print("Square completed - no turn needed")
            
            print(f"\n✓ Square {square + 1} completed!")
            
            # Pause between squares
            if square < num_squares - 1:
                print(f"Pausing before next square...")
                time.sleep(2)

    def test_movement(self):
        """Test basic movements"""
        print("\nTesting basic movements:")
        print("-" * 30)
        
        print("1. Forward 30cm...")
        self.move_straight(30)
        
        print("2. Turn right 90°...")
        self.turn_right_90()
        
        print("3. Forward 30cm...")
        self.move_straight(30)
        
        print("4. Turn right 90°...")
        self.turn_right_90()
        
        print("Test completed!")

    def adjust_parameters(self, move_speed=None, turn_speed=None, square_size=None):
        """Adjust movement parameters"""
        if move_speed:
            self.move_speed = move_speed
            print(f"Move speed adjusted to: {self.move_speed} m/s")
        
        if turn_speed:
            self.turn_speed = turn_speed  
            print(f"Turn speed adjusted to: {self.turn_speed} deg/s")
        
        if square_size:
            self.square_size = square_size
            print(f"Square size adjusted to: {self.square_size} cm")

    def cleanup(self):
        """Stop robot and close connection"""
        print("\nStopping robot and closing connection...")
        self.ep_chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.5)
        self.ep_robot.close()
        print("Connection closed successfully!")

def main():
    """Main execution function"""
    square_controller = None
    
    try:
        # Create controller
        square_controller = SquareMovementSimple()
        
        print("\nRoboMaster S1 Square Movement")
        print("Using drive_speed control only")
        print("=" * 40)
        
        # Option 1: Test basic movement first
        user_input = input("\nDo you want to test basic movements first? (y/n): ").lower()
        if user_input == 'y':
            square_controller.test_movement()
            time.sleep(2)
        
        # Option 2: Adjust parameters if needed
        user_input = input("\nDo you want to adjust parameters? (y/n): ").lower()
        if user_input == 'y':
            try:
                speed = float(input("Enter move speed (m/s, default 0.3): ") or 1.5)
                turn = float(input("Enter turn speed (deg/s, default 45): ") or 45)
                size = int(input("Enter square size (cm, default 100): ") or 0.6)
                square_controller.adjust_parameters(speed, turn, size)
            except ValueError:
                print("Invalid input, using default parameters")
        
        # Execute square movement
        try:
            num_squares = int(input("\nHow many squares to draw? (default 1): ") or 1)
        except ValueError:
            num_squares = 1
        
        print(f"\nStarting to draw {num_squares} square(s)...")
        time.sleep(2)  # Give user time to prepare
        
        square_controller.move_square(num_squares=num_squares)
        
        print("\n🎉 Mission completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        
    finally:
        # Always cleanup
        if square_controller:
            square_controller.cleanup()

if __name__ == "__main__":
    main()